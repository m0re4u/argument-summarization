import json

import numpy as np
from sklearn.metrics import average_precision_score

from argument_summarization.scorers.bart import BARTScorer
from argument_summarization.scorers.bleurt import BLEURTPyTorchScorer
from argument_summarization import get_project_root
from bert_score import BERTScorer


def parse_gpt_response(response_for_topic):
    """
    Parse GPT response for key points, popularity score and their stance.
    """
    raw_text = response_for_topic["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw_text)
    except json.decoder.JSONDecodeError:
        if "```" in raw_text:
            code_block = raw_text.split("```")[1]
            code_block = code_block.replace("json", "")
            try:
                parsed = json.loads(code_block)
            except json.decoder.JSONDecodeError:
                print(raw_text)
                raise ValueError(
                    "cannot parse response because I cannot interpret the code block"
                )
        else:
            print(raw_text)
            raise ValueError("cannot parse response because there is no code block")

    # Flatten list of key points (either contains a single list of key points or two of lists of key points, one for each stance)
    kpa = [x for k in parsed for x in parsed[k]]

    renamed_kps = []
    for kp in kpa:
        renamed_kps.append(
            {
                "keypoint": kp["reason"],
                "popularity": kp["popularity"],
                "stance": kp["stance"],
            }
        )

    return renamed_kps


def parse_smatch_response(response_for_topic):
    """
    Parse structured responses from the SMatchtoPR approach.
    """
    output = []
    for key_point in response_for_topic:
        output.append(
            {
                "keypoint": key_point,
                "stance": response_for_topic[key_point][0]["stance"],
                "num_matches": len(response_for_topic[key_point]),
            }
        )
    # sort output by number of matches and assign popularity score
    output = sorted(output, key=lambda x: x["num_matches"], reverse=True)
    for i, kp in enumerate(output):
        kp["popularity"] = i
    return output


def parse_kps_api_response(response_for_topic):
    """
    Parse structured responses from the IBM KPA API.
    """
    return [k for k in response_for_topic["keypoint_matchings"]]


def bleurt_precision(scorer, predictions, references):
    """
    BLEURT precision scores from https://arxiv.org/pdf/2305.16000.pdf
    """
    all_scores = []
    for prediction, all_refs in zip(predictions, references):
        scores = np.array(
            [
                scorer.score(references=[ref], candidates=[prediction])
                for ref in all_refs
            ]
        )
        all_scores.append(scores.max())
    return np.average(all_scores)


def bleurt_recall(scorer, predictions, references):
    """
    BLEURT recall scores from https://arxiv.org/pdf/2305.16000.pdf
    """
    all_refs = references[0]
    all_scores = []
    for ref in all_refs:
        scores = np.array(
            [scorer.score(references=[cand], candidates=[ref]) for cand in predictions]
        )
        all_scores.append(scores.max())
    return np.average(all_scores)


def compute_f1(precision, recall):
    """
    Compute F1 score from precision and recall.
    """
    return (2 * float(precision * recall)) / float(precision + recall)


def evaluate_bleurtscore(predictions, references):
    """
    Get all BLEURT scores for a given set of predictions and references.
    """
    scorer = BLEURTPyTorchScorer()
    soft_precision = bleurt_precision(scorer, predictions, references)
    soft_recall = bleurt_recall(scorer, predictions, references)
    soft_f1 = compute_f1(soft_precision, soft_recall)
    return soft_precision, soft_recall, soft_f1


def bart_precision(scorer, predictions, references):
    precisions = scorer.multi_ref_score(
        predictions, references, agg="max", batch_size=4
    )
    return np.tanh(np.exp((np.average(precisions)) / 2 + 1.3))


def bart_recall(scorer, predictions, references):
    all_refs = references[0]
    candidates_copied = [predictions for _ in all_refs]
    return bart_precision(scorer, all_refs, candidates_copied)


def evaluate_bartscore(predictions, references):
    bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    bart_scorer.load(path=get_project_root() / "models/bartscore/bart_score.pth")

    soft_precision = bart_precision(bart_scorer, predictions, references)
    soft_recall = bart_recall(bart_scorer, predictions, references)
    soft_f1 = compute_f1(soft_precision, soft_recall)
    return soft_precision, soft_recall, soft_f1


def evaluate_bertscore(predictions, references):
    bert_scorer = BERTScorer(lang="en", device="cuda:0", rescale_with_baseline=True)
    p, r, f1 = bert_scorer.score(predictions, references)

    soft_precision = p.mean().item()
    soft_recall = r.mean().item()
    soft_f1 = f1.mean().item()
    return soft_precision, soft_recall, soft_f1


# KPM Evaluation -------------


def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df) * top_percentile)
    df = df.sort_values("score", ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df["key_point_id"] == "dummy_id", "score"] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])


def calc_mean_average_precision(df, label_column):
    precisions = [
        get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])
    ]
    return np.mean(precisions)


def evaluate_predictions(merged_df):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    return mAP_strict, mAP_relaxed
