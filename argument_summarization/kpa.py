import json
import os
import uuid
import warnings
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import openai
import pandas as pd
import spacy
import tiktoken
import tqdm
from debater_python_api.api.clients.keypoints_client import KpsClient
from debater_python_api.api.debater_api import DebaterApi
from dotenv import load_dotenv
from fast_pagerank import pagerank
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT_CHATGPT_CLOSED_BOOK = """
Give me a JSON object of key arguments for and against the claim: {}. Make sure the reasons start with addressing the main point. Indicate per reason whether it supports (pro) or opposes (con) the claim. Rank all reasons from most to least popular. Make sure you generate a valid JSON object. The object should contain a list of dicts containing fields: 'reason' (str), 'popularity' (int), 'stance' (str).
"""

PROMPT_CHATGPT_OPEN_BOOK = """
Extract key arguments for and against the claim: {}. You need to extract the key arguments from the comments listed here: {}
Give me a JSON object of key arguments for and against the claim. Make sure the reasons start with addressing the main point. Indicate per reason whether it supports (pro) or opposes (con) the claim. Rank all reasons from most to least popular. Make sure you generate a valid JSON object. The object should contain a list of dicts containing fields: 'reason' (str), 'popularity' (int), 'stance' (str).
"""
MAX_COMMENTS_CHATGPT_OPEN_BOOK = {
    "kpa2021": 600,
    "hyena": 100,
    "thinkp": 100,
    "perspectrum": 100,
}


class KPAMethod:
    name: str

    def __init__(self, outdir: str = "results"):
        self.outdir = Path(outdir)

    def __call__(self, dataset):
        """
        Perform Key Point Analysis on the provided dataset. This is mostly used for generating
        results for the KPG task, not for the KPM task.
        """
        raise NotImplementedError

    def write_results(self, dataset_name, topic, results):
        """
        Write results from KPA to file.
        """
        raise NotImplementedError

    def predict_batch(self, df):
        """
        Predict the match labels for argument and key points for a single topic. This is used for
        the KPM task.
        """
        raise NotImplementedError

    def result_file_name(self, dataset, topic):
        """
        Create the filename for a result file.
        """
        return (
            self.outdir
            / self.name
            / dataset.dataset_name
            / f"{self.name}_{topic.replace(' ', '_').lower()}.json"
        )


class ChatGPTKPA(KPAMethod):
    name: str = "chatgpt_api"

    def __init__(self, outdir: str = "results", closed_book=True):
        self.prompt_type = "closed-book" if closed_book else "open-book"
        self.name = f"{self.name}_{self.prompt_type.replace('-', '_')}"
        (
            self.model_name,
            self.price_input,
            self.price_output,
            self.max_tokens,
        ) = self.setup_model()
        super().__init__(outdir)

    def write_results(self, dataset, topic, results):
        outfile = self.result_file_name(dataset, topic)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(results, f)

    def construct_prompts(self, dataset):
        """
        Construct the prompts depending on the type of prompting (closed/open).
        """
        prompts = {}
        for topic in dataset.get_topics():
            if self.prompt_type == "closed-book":
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": PROMPT_CHATGPT_CLOSED_BOOK.format(topic),
                    },
                ]

            elif self.prompt_type == "open-book":
                comments = dataset.get_comments(with_topic=topic)
                if len(comments) > MAX_COMMENTS_CHATGPT_OPEN_BOOK[dataset.dataset_name]:
                    np.random.shuffle(comments)
                    comments = comments[
                        : MAX_COMMENTS_CHATGPT_OPEN_BOOK[dataset.dataset_name]
                    ]

                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": PROMPT_CHATGPT_OPEN_BOOK.format(topic, comments),
                    },
                ]
            prompts[topic] = message
        return prompts

    def setup_model(self):
        """
        Setup pricing/num tokens depending on how we want to run the model.
        """
        if self.prompt_type == "closed-book":
            model = "gpt-3.5-turbo"
            price_input = 0.0000015
            price_output = 0.000002
            max_tokens = 4096
        elif self.prompt_type == "open-book":
            model = "gpt-3.5-turbo-16k"
            price_input = 0.000003
            price_output = 0.000004
            max_tokens = 16384
        else:
            raise ValueError(f"Unknown prompt type {self.prompt_type}")
        return model, price_input, price_output, max_tokens

    def measure_tokens(self, prompts):
        """
        Measure the number of tokens in the prompts. Probably underestimates due to ignoring the
        user/system message indicator.
        """
        enc = tiktoken.encoding_for_model(self.model_name)
        tokenized_prompts = {
            topic: enc.encode("\n".join([x["content"] for x in message]))
            for topic, message in prompts.items()
        }
        return tokenized_prompts, {t: len(x) for t, x in tokenized_prompts.items()}

    def measure_tokens_and_cost(self, prompts, only_cost=False):
        """
        Measure the number of tokens in the prompts and the cost of running the model.
        """
        if only_cost:
            num_tokens_per_prompt = {
                f"dummy_{i}": len(x) for i, x in enumerate(prompts)
            }
        else:
            _, num_tokens_per_prompt = self.measure_tokens(prompts)
            for topic, num_tokens in num_tokens_per_prompt.items():
                if num_tokens > self.max_tokens:
                    print(
                        f"Number of tokens for topic {topic} is {num_tokens} which is larger than the max tokens {self.max_tokens}, removing.."
                    )
                    prompts.pop(topic)

        num_tokens = sum(num_tokens_per_prompt.values())
        return (
            num_tokens,
            num_tokens * self.price_input,
            sum(
                [
                    ((self.max_tokens - x) * self.price_output) + (x * self.price_input)
                    for x in num_tokens_per_prompt.values()
                ]
            ),
        )

    def parse_batch_response(self, all_completions):
        """
        Parse the output from ChatGPT of argument/key point match labels
        """
        scores = []
        arg_ids = []
        arg_id2score = {}
        for compl in all_completions:
            response_raw = compl["choices"][0]["message"]["content"]

            try:
                response = json.loads(response_raw)
            except json.decoder.JSONDecodeError:
                # print(response_raw)
                raise
            # sometimes GPT borkes up, correctly interpret it
            if "arguments" in response:
                try:
                    first_item = response["arguments"][0]
                    if "ID" in first_item:
                        response = {
                            str(d["ID"]): d["match"] for d in response["arguments"]
                        }
                    elif "id" in first_item:
                        response = {
                            str(d["id"]): d["match"] for d in response["arguments"]
                        }
                except KeyError:
                    # print(response)
                    continue
            for arg_id, match_label in response.items():
                if "ID:" in arg_id:
                    arg_id = int(arg_id.replace("ID:", ""))

                if isinstance(match_label, dict):
                    match_label = match_label["match"]

                if match_label:
                    score = 1

                else:
                    score = 0

                scores.append(score)

                arg_ids.append(arg_id)
                arg_id2score[str(arg_id)] = score
        print(f"Number of scores: {len(scores)}")
        return scores, arg_id2score

    def create_prompts_for_batch_tokens(
        self, df, enc, prompt_start, answer_padding=3000
    ):
        prompts = []
        tokenized_prompts = []
        prompt_ids = []
        argids = []
        prompt = f"{prompt_start}"

        for i, row in df.iterrows():
            # TODO: restrict argument length for perspectrum
            new_str = f"\nID: {i} nArgument: {row['argument']} Key point: {row['key_point']}\n"
            encoded_prompt = enc.encode(prompt)
            encoded_new = enc.encode(new_str)
            if len(encoded_prompt) + len(encoded_new) > (
                self.max_tokens - answer_padding
            ):
                prompts.append(prompt)
                tokenized_prompts.append(encoded_prompt)
                prompt_ids.append(argids)
                argids = []
                prompt = prompt_start
            prompt += new_str
            argids.append(i)

        prompts.append(prompt)
        tokenized_prompts.append(enc.encode(prompt))
        prompt_ids.append(argids)
        return prompts, tokenized_prompts, prompt_ids

    def create_prompts_for_batch_count(
        self, df, enc, prompt_start, num_args_per_prompt=10
    ):
        prompts = []
        tokenized_prompts = []
        prompt_ids = []
        argids = []
        prompt = f"{prompt_start}"
        arg_counter = 0

        df.reset_index(inplace=True)

        for i, row in df.iterrows():
            new_str = f"\nID: {i} nArgument: {row['argument']} Key point: {row['key_point']}\n"
            if arg_counter > num_args_per_prompt:
                prompts.append(prompt)
                encoded_prompt = enc.encode(prompt)
                tokenized_prompts.append(encoded_prompt)
                prompt_ids.append(argids)
                argids = []
                arg_counter = 0
                prompt = prompt_start
            prompt += new_str
            argids.append(i)
            arg_counter += 1

        prompts.append(prompt)
        tokenized_prompts.append(enc.encode(prompt))
        prompt_ids.append(argids)
        return prompts, tokenized_prompts, prompt_ids

    def predict_batch(self, df, prompt_construction_strategy="count"):
        """
        Predict the match labels for argument and key points for a single topic
        """

        topic = df["topic"].iloc[0]
        topic = re.sub(r"[^a-zA-Z\d\s:]+", "", topic).lower()
        PROMPT_START = f'For the topic of {topic}, indicate for each of the following argument/key point pairs whether the argument matches the key point. Return a json object with just a "match" boolean per argument/key point pair.'

        # Hack to give the dataset a dataset_name attribute
        TemporaryDataset = namedtuple("TemporaryDataset", "dataset_name")
        dataset = TemporaryDataset(dataset_name=df["dataset"].iloc[0])

        enc = tiktoken.encoding_for_model(self.model_name)
        if prompt_construction_strategy == "token":
            (
                prompts,
                tokenized_prompts,
                prompt_arg_ids,
            ) = self.create_prompts_for_batch_tokens(df, enc, PROMPT_START)
        elif prompt_construction_strategy == "count":
            (
                prompts,
                tokenized_prompts,
                prompt_arg_ids,
            ) = self.create_prompts_for_batch_count(df, enc, PROMPT_START)
        else:
            raise ValueError(
                f"unknown prompt construction strategy {prompt_construction_strategy}"
            )
        print()
        print(
            f"Dataset {dataset.dataset_name} - Topic: {topic} - {len(prompt_arg_ids)}"
        )
        print(f"Number of arg/kp pairs: {len(df)}")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Number of tokens: {sum([len(x) for x in tokenized_prompts])}")
        tokens, min_cost, max_cost = self.measure_tokens_and_cost(
            tokenized_prompts, only_cost=True
        )
        print(
            f"Number of input tokens: {tokens} and min cost: ${min_cost:.2f} with a max of ${max_cost:.2f}"
        )
        print(f"Do you want to continue? (y/n)")
        if input() != "y":
            return

        if self.result_file_name(dataset, topic).exists():
            # load chat completions from file
            print("Exists! im loading the results from file :) ")
            with open(self.result_file_name(dataset, topic), "r") as f:
                all_completions = json.load(f)
        else:
            all_completions = []
            for prompt in tqdm.tqdm(prompts):
                chat_completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                all_completions.append(chat_completion)

        self.write_results(dataset, topic, all_completions)
        scores, mapped_args = self.parse_batch_response(all_completions)

        for i, _ in df.iterrows():
            if str(i) not in mapped_args:
                print(i)
                # insert a zero score at i
                scores.insert(i, 0.0)
        assert len(scores) == len(df), f"{len(scores)} != {len(df)}"
        return scores

    def __call__(self, dataset):
        all_results = {}

        # Create the prompts we want to run
        prompts = self.construct_prompts(dataset)
        print(
            f"[{self.name}] Constructed prompts for dataset {dataset.dataset_name} with {len(prompts)} topics"
        )
        tokens, min_cost, max_cost = self.measure_tokens_and_cost(prompts)
        print(
            f"Number of input tokens: {tokens} and min cost: ${min_cost:.2f} with a max of ${max_cost:.2f}"
        )
        print(f"Do you want to continue? (y/n)")
        if input() != "y":
            return

        for topic, prompt in tqdm.tqdm(prompts.items()):
            if self.result_file_name(dataset, topic).exists():
                continue

            chat_completion = openai.ChatCompletion.create(
                model=self.model_name, messages=prompt
            )
            all_results[topic] = chat_completion
            self.write_results(dataset, topic, chat_completion)
        return all_results


class OpenLLMKPA(ChatGPTKPA):
    name: str = "openllm_api"
    POSSIBLE_MODELS = []

    def __init__(self, model_name, outdir: str = "results", closed_book=True):
        self.prompt_type = "closed-book" if closed_book else "open-book"
        assert model_name in self.POSSIBLE_MODELS, f"Unknown model {model_name}"
        self.model_name = model_name
        self.name = (
            f"{self.name}_{self.model_name}_{self.prompt_type.replace('-', '_')}"
        )
        super().__init__(outdir)

    def req_completion(self, prompt):
        pass

    def __call__(self, dataset):
        all_results = {}

        # Create the prompts we want to run
        prompts = self.construct_prompts(dataset)
        print(
            f"[{self.name}] Constructed prompts for dataset {dataset.dataset_name} with {len(prompts)} topics"
        )
        for topic, prompt in tqdm.tqdm(prompts.items()):
            if self.result_file_name(dataset, topic).exists():
                continue

            chat_completion = self.req_completion(prompt)
            all_results[topic] = chat_completion
            self.write_results(dataset, topic, chat_completion)
        return all_results


class KPS_APIKPA(KPAMethod):
    name: str = "kps_api"

    def write_results(self, dataset, topic, results):
        outfile = self.result_file_name(dataset, topic)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(results, dict):
            with open(outfile, "w") as f:
                json.dump(results, f)
        else:
            results.save(str(outfile))

    def predict_single(self, topic, argument, key_point):
        debater_api = DebaterApi(os.getenv("IBM_API_KEY"))
        keypoints_client = debater_api.get_keypoints_client()
        KpsClient.init_logger()
        keypoints_client.create_domain(topic, ignore_exists=True)
        comments_ids = [str(uuid.uuid4()).replace("-", "_")]
        keypoints_client.upload_comments(topic, comments_ids, [argument])
        try:
            keypoint_matching = keypoints_client.run_kps_job(
                topic,
                run_params={
                    "key_points": [key_point],
                    "min_matches_per_kp": 0,
                    "mapping_policy": "LOOSE",
                },
                stance="no-stance",
            )
            try:
                score = keypoint_matching.result_df.iloc[0].match_score
                print(f"Got score: {score}")
            except:
                print(keypoint_matching.result_df)
                score = 0
            return score
        except Exception as e:
            print(f"Failed to run KPS for topic {topic} with error {e}")
            raise

    def predict_batch(self, df):
        """
        Get the key point matches for a batch of data for a single topic.

        Return the scores for each argument-key point pair.
        """
        debater_api = DebaterApi(os.getenv("IBM_API_KEY"))
        keypoints_client = debater_api.get_keypoints_client()
        keypoints_client.delete_all_domains_cannot_be_undone()
        topic = df.topic.iloc[0]
        # remove non-alphanumeric characters
        topic = re.sub(r"\W+", "", topic)
        keypoints_client.create_domain(topic, ignore_exists=False)

        # Upload all arguments
        if "arg_id" in df.columns:
            comment_ids = df.arg_id.unique().tolist()
        else:
            comment_ids = [str(i) for i, _ in enumerate(df.argument.unique().tolist())]

        comments = [
            x[:2999] if len(x) >= 3000 else x for x in df.argument.unique().tolist()
        ]
        keypoints_client.upload_comments(
            topic,
            comment_ids,
            comments,
        )
        try:
            keypoint_matching = keypoints_client.run_kps_job(
                topic,
                run_params={
                    "key_points": df.key_point.unique().tolist(),
                    "min_matches_per_kp": 0,
                    "mapping_policy": "LOOSE",
                },
                stance="no-stance",
            )
            result_df = keypoint_matching.result_df
            # Match result df back to the rows in the df. If a row is missing, it means that the
            # argument was not matched
            out_df = df.copy()
            out_df["score"] = np.nan
            for i, row in result_df.iterrows():
                x = out_df.argument == row.sentence_text
                y = out_df.key_point == row.kp
                out_df.loc[x & y, "score"] = row.match_score

            out_df.score.fillna(0, inplace=True)
            return out_df.score.tolist()
        except Exception as e:
            print(f"Failed to run KPS for topic {topic} with error ({type(e)}) {e}")
            raise

    def __call__(self, dataset, with_stance="each-stance"):
        debater_api = DebaterApi(os.getenv("IBM_API_KEY"))
        keypoints_client = debater_api.get_keypoints_client()
        KpsClient.init_logger()

        all_results = {}
        for topic in tqdm.tqdm(dataset.get_topics()):
            if self.result_file_name(dataset, topic).exists():
                continue

            comments = dataset.get_comments(with_topic=topic)
            try:
                keypoint_matchings_result = keypoints_client.run_full_kps_flow(
                    domain=topic,
                    comments_texts=comments,
                    stance=with_stance,
                    run_params={
                        "mapping_policy": "LOOSE",
                        "kp_granularity": "FINE",
                        "kp_relative_aq_threshold": 0.5,
                        "kp_min_len": 0,
                        "kp_max_len": 100,
                        "kp_min_kp_quality": 0.5,
                    },
                )
                all_results[topic] = keypoint_matchings_result
                self.write_results(dataset, topic, keypoint_matchings_result)
            except Exception as e:
                print(f"Failed to run KPS for topic {topic} with error {e}")
                self.write_results(dataset, topic, {"error_message": str(e)})
                continue

        return all_results


class SMatchToPRKPA(KPAMethod):
    name: str = "smatch_to_pr"

    def __init__(self, load_model, outdir="results", create_new_kps=False):
        self.name = "smatch_to_pr"
        self.model_name = load_model
        self.model = SentenceTransformer(self.model_name)
        self.create_new_keypoints = create_new_kps
        self.debater_api = DebaterApi(os.getenv("IBM_API_KEY"))
        self.arg_quality_client = self.debater_api.get_argument_quality_client()
        self.page_rank_p = 0.2
        self.page_rank_min_quality_score = 0.8
        self.page_rank_min_match_score = 0.8
        self.page_rank_min_length = 5
        self.page_rank_max_length = 20
        self.filter_min_match_score = 0.8
        self.filter_min_result_length = 5
        self.filter_timeout = 1000
        super().__init__(outdir)

    def get_candidates(self, topic, sentences):
        sentence_topic_dicts = [
            {"sentence": sentence, "topic": topic} for sentence in sentences
        ]
        return sentence_topic_dicts

    @lru_cache(maxsize=1000000)
    def call_argq_ibm_api(self, sentence_data):
        scores = self.arg_quality_client.run(
            [{"sentence": x[0], "topic": x[1]} for x in sentence_data]
        )
        return scores

    def score_candidates(self, candidates):
        # Flatten candidates and keep track of which candidate belongs to which argument
        c = []
        c_idx = []
        for i, x in enumerate(candidates):
            for y in x:
                c.append(y)
                c_idx.append(i)
        # Get arg scores
        # create tuple of sentence and topic
        c = [(x["sentence"], x["topic"]) for x in c]
        scores = self.call_argq_ibm_api(tuple(c))

        # Group scores by argument
        scored_candidates = pd.Series(
            [[] for i in range(len(candidates))], index=candidates.index
        )
        score_counter = 0
        for idx, x in candidates.items():
            for y in x:
                scored_candidates[idx].append((y["sentence"], scores[score_counter]))
                score_counter += 1

        return scored_candidates

    def gen_match_matrix(self, sents, topic):
        sents1 = [topic + " <SEP> " + x for x in sents]
        sents1_embeddings = self.model.encode(sents1)

        sim_matrix = cosine_similarity(sents1_embeddings, sents1_embeddings)
        super_threshold_indices = sim_matrix < self.page_rank_min_match_score
        sim_matrix[super_threshold_indices] = 0

        return sim_matrix

    def apply_page_rank(self, row):
        cand_sents = [
            x
            for x in row["sents_with_scores"]
            if x[1] > self.page_rank_min_quality_score
            and len(x[0].split()) < self.page_rank_max_length
            and len(x[0].split()) > self.page_rank_min_length
        ]
        if len(cand_sents) == 0:
            return []
        cands, cands_qualities = zip(*cand_sents)
        cands_matching_mat = self.gen_match_matrix(cands, row["topic"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = pagerank(
                cands_matching_mat,
                personalize=np.array(cands_qualities),
                p=self.page_rank_p,
            )
        ranked_candidates = list(zip(cands, pr))
        return sorted(ranked_candidates, key=lambda x: -x[1])

    def filter_ranked_list(self, row):
        ranked_sents = [x[0] for x in row["ranked_sents"]]
        filtered_sents = []
        min_match = self.filter_min_match_score
        tries = 0
        while (
            len(filtered_sents) < self.filter_min_result_length
            and self.filter_timeout > tries
        ):
            tries += 1
            filtered_sents = []
            for i, s in enumerate(ranked_sents):
                if len(filtered_sents) == 0:
                    filtered_sents.append(s)
                else:
                    matching_scores = self.gen_match_matrix(
                        [s] + filtered_sents, row["topic"]
                    )
                    max_sim = np.max(matching_scores[0][1:])
                    if max_sim < min_match:
                        filtered_sents.append(s)

            min_match = min_match + 0.1

        return filtered_sents

    def generate_kps(self, df):
        sent_split = English()
        sent_split.add_pipe("sentencizer")
        nlp = spacy.load("en_core_web_sm")

        df["sentences"] = df.argument.apply(
            lambda x: [
                x.text
                for x in sent_split(x).sents
                if list(nlp(x.text))[0].pos_ != "PRON"
            ]
        )

        candidates = df.apply(
            lambda row: self.get_candidates(row["topic"], row["sentences"]),
            axis=1,
        )
        df["sents_with_scores"] = self.score_candidates(candidates)
        evaluation_df = (
            df.groupby(["topic", "stance"])
            .agg(
                {
                    "sents_with_scores": lambda x: set(
                        [item for items in x for item in items]
                    )
                }
            )
            .reset_index()
        )
        evaluation_df["ranked_sents"] = evaluation_df.apply(
            lambda row: self.apply_page_rank(row),
            axis=1,
        )

        evaluation_df["ranked_kps"] = evaluation_df.apply(
            lambda row: self.filter_ranked_list(row), axis=1
        )
        return evaluation_df["ranked_kps"][0][:5]

    def write_results(self, dataset, topic, results):
        outfile = self.result_file_name(dataset, topic)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(results, f)

    def match_argument_with_keypoints(self, kp_embeddings, argument_embeddings):
        arg2kp = {}
        cosine_scores = np.zeros((len(argument_embeddings), len(kp_embeddings)))
        for arg_i, arg_embedding in enumerate(argument_embeddings):
            for kp_i, kp_embedding in enumerate(kp_embeddings):
                cosine_scores[arg_i][kp_i] = util.pytorch_cos_sim(
                    arg_embedding, kp_embedding
                ).item()

            # Get the argmax of the cosine similarities
            kp_scores = np.array(cosine_scores[arg_i])
            arg2kp[arg_i] = {"key_point": kp_scores.argmax(), "score": kp_scores.max()}

        return arg2kp

    def predict_batch(self, df):
        kp_embedding = self.model.encode(df["key_point"].tolist())
        arg_embedding = self.model.encode(df["argument"].tolist())
        return util.pytorch_cos_sim(arg_embedding, kp_embedding).diag().tolist()

    def __call__(self, dataset):
        all_results = {}
        for topic in dataset.get_topics():
            # Check if result already exists and skip if so
            if self.result_file_name(dataset, topic).exists():
                continue

            all_results[topic] = defaultdict(list)
            for stance in ["con", "pro"]:
                df = dataset.get_comments(
                    with_topic=topic, with_stance=stance, all_data=True
                )
                df = df.dropna()
                if len(df) == 0:
                    print(f"Empty dataframe for topic {topic}/{stance}")
                    continue

                if self.create_new_keypoints:
                    # Generate new key points
                    key_points = self.generate_kps(df)
                else:
                    # Take key points from data
                    key_points = df["key_point"].unique().tolist()

                if len(key_points) == 0:
                    print(
                        f"failed to generate any key points for topic {topic}/{stance} out of {len(df)} comments.."
                    )
                    continue
                arguments = df["argument"].unique().tolist()

                topic_keypoints_embeddings = self.model.encode(key_points)
                topic_arguments_embeddings = self.model.encode(arguments)
                arg2kp = self.match_argument_with_keypoints(
                    topic_keypoints_embeddings,
                    topic_arguments_embeddings,
                )
                for arg_id, assignment in arg2kp.items():
                    arg = arguments[arg_id]
                    kp = key_points[assignment["key_point"]]
                    score = assignment["score"]
                    all_results[topic][kp].append(
                        {
                            "argument": arg,
                            "score": score,
                            "stance": stance,
                        }
                    )

            self.write_results(dataset, topic, all_results[topic])
        return all_results
