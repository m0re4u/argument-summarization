import argparse
import json
from pathlib import Path

import pandas as pd
import tqdm
from logging import getLogger
from sacrerouge.metrics import Rouge

# Disable logging from transformers
logger = getLogger("transformers")
logger.setLevel(50)


from argument_summarization.data import (
    HyEnADataset,
    KPA2021Dataset,
    PerspectrumDataset,
)
from argument_summarization.metrics import (
    evaluate_bertscore,
    evaluate_bartscore,
    evaluate_bleurtscore,
    parse_gpt_response,
    parse_kps_api_response,
    parse_smatch_response,
)


def compare_kps(pred_kps, true_kps):
    """
    Seq2seq evaluation of key point strings.

    Returns the following metrics:
    - ROUGE-1 (precision, recall, f1)
    - ROUGE-2 (precision, recall, f1)
    - ROUGE-L (precision, recall, f1)
    - Soft-precision with BERTScore
    - Soft-recall with BERTScore
    - Soft-F1 with BERTScore
    - Soft-precision with BLEURT
    - Soft-recall with BLEURT
    - Soft-F1 with BLEURT
    - Soft-precision with BARTScore
    - Soft-recall with BARTScore
    - Soft-F1 with BARTScore
    """
    rouge = Rouge(compute_rouge_l=True, max_ngram=2, scoring_function="max")
    scores = []

    if len(pred_kps) == 0 or len(true_kps) == 0:
        return [], {}

    # Compare all key points with all references since we don't know which key point the predicted
    # ones match to. Take the max overlapping key point as the reference and final score.
    for kp in pred_kps:
        res = rouge.score(kp, true_kps)
        res_flat = res.flatten_keys()
        scores.append(res_flat)

    be_p, be_r, be_f1 = evaluate_bertscore(
        predictions=pred_kps, references=[true_kps for _ in pred_kps]
    )

    ba_p, ba_r, ba_f1 = evaluate_bartscore(
        predictions=pred_kps, references=[true_kps for _ in pred_kps]
    )
    bl_p, bl_r, bl_f1 = evaluate_bleurtscore(
        predictions=pred_kps, references=[true_kps for _ in pred_kps]
    )
    semantic_eval = {
        "bert_recall": be_r,
        "bert_precision": be_p,
        "bert_f1": be_f1,
        "bart_recall": ba_r,
        "bart_precision": ba_p,
        "bart_f1": ba_f1,
        "bleurt_recall": bl_r,
        "bleurt_precision": bl_p,
        "bleurt_f1": bl_f1,
    }
    return scores, semantic_eval


def agg_results(rouge_results, semantic_results):
    """
    Aggregate results from a single topic and stance combination into a single dataframe and
    compute descriptive statistics.
    """
    # Mean over individual KP similarity scores
    rouge_results = pd.DataFrame.from_records(rouge_results)
    rouge_means = rouge_results.mean()

    # Mean over semantic similarity scores per topic
    semantic_results = pd.DataFrame.from_records(semantic_results)
    semantic_means = semantic_results.mean()
    return pd.concat([rouge_means, semantic_means])


def select_fraction_kps(all_data, fraction):
    counts = all_data["key_point"].value_counts(dropna=False)
    limit_idx = len(counts) * fraction
    # select those key points that are the least frequent
    key_points_by_frequency = list(reversed(counts.index))
    selected_key_points = key_points_by_frequency[: int(limit_idx)]
    return selected_key_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kp_cutoff",
        type=float,
        help="fraction of key points to take into consideration starting form the least frequent",
        default=-1,
    )
    parser.add_argument(
        "--focus_dataset",
        type=str,
        help="If not None, focus on a single dataset",
        default=None,
    )
    args = parser.parse_args()

    if args.focus_dataset is not None:
        if args.focus_dataset == "hyena":
            datasets = {"HyEnA": HyEnADataset("data/hyena/", split="test")}
        elif args.focus_dataset == "kpa2021":
            datasets = {"KPA2021": KPA2021Dataset("data/kpa2021/", split="test")}
        elif args.focus_dataset == "perspectrum":
            datasets = {
                "Perspectrum": PerspectrumDataset("data/perspectrum/", split="test")
            }
    else:
        datasets = {
            "KPA2021": KPA2021Dataset("data/kpa2021/", split="test"),
            "HyEnA": HyEnADataset("data/hyena/", split="test"),
            "Perspectrum": PerspectrumDataset("data/perspectrum/", split="test"),
        }

    methods = {
        "chatgpt_api_open_book": parse_gpt_response,
        "chatgpt_api_closed_book": parse_gpt_response,
        "kps_api": parse_kps_api_response,
        "smatch_to_pr": parse_smatch_response,
    }

    missing_results = []

    result_dump = {
        "cutoff": args.kp_cutoff,
        "focus_dataset": args.focus_dataset,
        "results": {},
    }

    for dataset_name, dataset in tqdm.tqdm(datasets.items()):
        rouge_results = {m: [] for m in methods}
        semantic_results = {m: [] for m in methods}
        if args.kp_cutoff >= 0.0:
            selected_key_points = select_fraction_kps(
                dataset.get_comments(all_data=True), args.kp_cutoff
            )

        for topic in tqdm.tqdm(dataset.get_topics(), leave=False, desc="Topics"):
            for method_name, method in tqdm.tqdm(
                methods.items(), leave=False, desc="Methods"
            ):
                method_dir = Path(f"results/{method_name}/{dataset_name.lower()}")
                result_file = (
                    method_dir / f"{method_name}_{topic.replace(' ', '_').lower()}.json"
                )
                if result_file.exists():
                    # Load predictions from file
                    with open(result_file) as f:
                        results = json.load(f)
                    # Parse results from file
                    try:
                        res = method(results)
                    except:
                        missing_results.append(
                            {
                                "topic": topic,
                                "method": method_name,
                                "dataset": dataset_name,
                            }
                        )

                    # Obtain ground truth data
                    ground_truth_data_pro = dataset.get_comments(
                        with_topic=topic, all_data=True, with_stance="pro"
                    )
                    ground_truth_data_con = dataset.get_comments(
                        with_topic=topic, all_data=True, with_stance="con"
                    )

                    if args.kp_cutoff >= 0.0:
                        ground_truth_data_pro = ground_truth_data_pro[
                            ground_truth_data_pro["key_point"].isin(selected_key_points)
                        ]
                        ground_truth_data_con = ground_truth_data_con[
                            ground_truth_data_con["key_point"].isin(selected_key_points)
                        ]

                    if (
                        len(ground_truth_data_pro) == 0
                        and len(ground_truth_data_con) == 0
                    ):
                        missing_results.append(
                            {
                                "topic": topic,
                                "method": method_name,
                                "dataset": dataset_name,
                            }
                        )

                    # Get evaluation metrics for pro
                    rouge_metrics_pro, semantic_pro = compare_kps(
                        [x["keypoint"] for x in res if x["stance"] == "pro"],
                        ground_truth_data_pro["key_point"].dropna().unique().tolist(),
                    )
                    # .. and for con
                    rouge_metrics_con, semantic_con = compare_kps(
                        [x["keypoint"] for x in res if x["stance"] == "con"],
                        ground_truth_data_con["key_point"].dropna().unique().tolist(),
                    )

                    # Writeback
                    rouge_results[method_name].extend(rouge_metrics_pro)
                    rouge_results[method_name].extend(rouge_metrics_con)
                    semantic_results[method_name].append(semantic_pro)
                    semantic_results[method_name].append(semantic_con)

                else:
                    missing_results.append(
                        {"topic": topic, "method": method_name, "dataset": dataset_name}
                    )

        # Output results
        if (
            len(rouge_results[list(methods.keys())[0]]) > 0
            and len(semantic_results[list(methods.keys())[0]]) > 0
        ):
            # print()
            for method_name, method in methods.items():
                aggd = agg_results(
                    rouge_results[method_name], semantic_results[method_name]
                )
                # print()
                # print()
                # print()
                # print(f"Results for {dataset_name} with {method_name}")
                # print(aggd)
                # print("====================================")
                result_dump["results"][method_name] = aggd.to_dict()
    # print("Missing results:")
    # for missing_result in missing_results:
    #     print(missing_result)

    with open(f"results/lower_limit_kpg/cutoff_results.json", "a") as f:
        json.dump(result_dump, f)
        f.write("\n")
