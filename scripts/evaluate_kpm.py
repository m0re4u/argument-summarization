import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from argument_summarization.data import HyEnADataset, KPA2021Dataset, PerspectrumDataset
from argument_summarization.kpa import KPS_APIKPA, ChatGPTKPA, SMatchToPRKPA


def get_evaluation_metric(
    df, label_column, top_percentile=0.5, metric="ap", cutoff=0.5
):
    """
    Get average precision for a dataframe
    """
    if len(df) < 10:
        top = len(df)
    else:
        top = int(len(df) * top_percentile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # interpret score column as float
        df["score"] = df["score"].astype(float)
        df = df.sort_values("score", ascending=False).head(top)
        if metric == "ap":
            score = average_precision_score(
                y_true=df[label_column], y_score=df["score"]
            )
        elif metric == "precision":
            y_pred = df["score"] > cutoff
            y_pred = y_pred.astype(float)
            score = precision_score(
                y_true=df[label_column], y_pred=y_pred, zero_division=0
            )
        elif metric == "recall":
            y_pred = df["score"] > cutoff
            y_pred = y_pred.astype(float)
            score = recall_score(
                y_true=df[label_column], y_pred=y_pred, zero_division=0
            )
        elif metric == "f1":
            y_pred = df["score"] > cutoff
            y_pred = y_pred.astype(float)
            score = f1_score(y_true=df[label_column], y_pred=y_pred, zero_division=0)
    return score


def get_scores(df, pretty_print=True, kp_comment_limit=None):
    """
    Get mean average precision scores for dataset and method combinations
    """
    scores = []
    for method, method_df in df.groupby("method"):
        for dataset, dataset_df in method_df.groupby("dataset"):
            results = []
            dataset_mean_score = dataset_df["score"].mean()
            for _, topic_df in dataset_df.groupby("topic"):
                # Get AP
                for k in [0.1, 0.2, 0.5, 0.8, 0.99]:
                    results.append(
                        {
                            f"ap (top_pct={k})": get_evaluation_metric(
                                topic_df, "label", top_percentile=k
                            ),
                        }
                    )
                cutoffs = np.arange(0, 1, 0.1).tolist() + [dataset_mean_score]
                for c in cutoffs:
                    results.append(
                        {
                            f"precision (top_pct=1, cutoff={c})": get_evaluation_metric(
                                topic_df,
                                "label",
                                top_percentile=1.0,
                                metric="precision",
                                cutoff=c,
                            ),
                            f"recall (top_pct=1, cutoff={c})": get_evaluation_metric(
                                topic_df,
                                "label",
                                top_percentile=1.0,
                                metric="recall",
                                cutoff=c,
                            ),
                            f"f1 (top_pct=1, cutoff={c})": get_evaluation_metric(
                                topic_df,
                                "label",
                                top_percentile=1.0,
                                metric="f1",
                                cutoff=c,
                            ),
                        }
                    )
            res = pd.DataFrame(results)
            res_mean = res.mean()
            if pretty_print:
                print(
                    f"Method: {method} Dataset: {dataset} AP: {res_mean['ap (top_pct=0.5)']:.4f}"
                )
            else:
                print(
                    f"{kp_comment_limit},{method},{dataset},{res_mean['ap (top_pct=0.5)']}"
                )
            scores.append((method, dataset, res))


def tmp_res_filename(method_name, dataset_name, topic):
    return f"tmp_results/{method_name}_{dataset_name}_{topic}.jsonl"


def get_topic_results(dataset_name, method, topic_df, gather_new=False):
    topic_results = []
    if Path(tmp_res_filename(method.name, dataset_name, topic)).exists():
        with open(
            tmp_res_filename(method.name, dataset_name, topic),
            "r",
        ) as f:
            for line in f.readlines():
                data = json.loads(line)
                if data["key_point"] in topic_df["key_point"].values:
                    topic_results.append(data)
    else:
        if not gather_new:
            return []
        topic_df["dataset"] = dataset_name
        scores = method.predict_batch(topic_df)
        topic_df.reset_index(inplace=True)
        for i, row in tqdm.tqdm(
            topic_df.iterrows(),
            desc="Rows: ",
            leave=False,
            total=len(topic_df),
        ):
            topic_results.append(
                {
                    "dataset": dataset_name,
                    "method": method.name,
                    "topic": topic,
                    "argument": row["argument"],
                    "key_point": row["key_point"],
                    "score": scores[i],
                    "label": row["label"],
                }
            )
        outpath = Path(tmp_res_filename(method.name, dataset.dataset_name, topic))
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            for record in topic_results:
                f.write(json.dumps(record) + "\n")

    return topic_results


def limit_data(topic_df, cutoff):
    """
    Limit a dataset to only contain those key points that have at most cutoff
    comments matched to them.
    """
    topic_df = topic_df[topic_df["label"] == 1]
    counts = topic_df["key_point"].value_counts(dropna=False)
    valids = counts[counts <= cutoff].index
    keep = topic_df[topic_df["key_point"].isin(valids)]
    return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kp_comment_cutoff",
        type=int,
        help="Maximum number of comments per key point for taking the kp into consideration",
        default=-1,
    )
    parser.add_argument(
        "--focus_dataset",
        type=str,
        help="If not None, focus on a single dataset",
        default=None,
    )
    parser.add_argument(
        "--gather_new",
        action="store_true",
        help="If KPM results do not exist, get them",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="If applicable, the directory of the (trained) model to use",
        default=None,
    )

    args = parser.parse_args()

    # Load data
    if args.focus_dataset == "kpa2021":
        datasets = [
            KPA2021Dataset("data/kpa2021/", split="test"),
        ]
    elif args.focus_dataset == "hyena":
        datasets = [
            HyEnADataset("data/hyena/", split="test"),
        ]
    elif args.focus_dataset == "perspectrum":
        datasets = [
            PerspectrumDataset(
                "data/perspectrum/", split="test", limit_topics_frac=1.0
            ),
        ]
    elif args.focus_dataset == "perspectrum-idebate":
        datasets = [
            PerspectrumDataset(
                "data/perspectrum/", split="test", limit_source="idebate"
            ),
        ]
    elif args.focus_dataset == "perspectrum-procon":
        datasets = [
            PerspectrumDataset(
                "data/perspectrum/", split="test", limit_source="procon"
            ),
        ]
    elif args.focus_dataset == "perspectrum-debatewise":
        datasets = [
            PerspectrumDataset(
                "data/perspectrum/", split="test", limit_source="debatewise"
            ),
        ]
    else:
        datasets = [
            KPA2021Dataset("data/kpa2021/", split="test"),
            HyEnADataset("data/hyena/", split="test"),
            PerspectrumDataset(
                "data/perspectrum/", split="test", limit_topics_frac=1.0
            ),
        ]

    # Load methods
    methods = [
        ChatGPTKPA(outdir="results_kpm", closed_book=False),
        KPS_APIKPA(outdir="results_kpm"),
        SMatchToPRKPA(
            args.model_dir,
            outdir="results_kpm",
            create_new_kps=False,
        ),
    ]

    records = []
    for dataset in tqdm.tqdm(datasets, desc="Datasets", leave=False):
        for topic, topic_data in tqdm.tqdm(
            dataset.kpm_data.groupby("topic"), desc="Topics: ", leave=False
        ):
            if args.kp_comment_cutoff > 0:
                topic_data = limit_data(topic_data, args.kp_comment_cutoff)
            for method in tqdm.tqdm(methods, desc="Methods: ", leave=False):
                topic_results = get_topic_results(
                    dataset.dataset_name,
                    method,
                    topic_data,
                    gather_new=args.gather_new,
                )

                records.extend(topic_results)
    df = pd.DataFrame(records)
    if len(df) > 0:
        if args.kp_comment_cutoff > 0:
            scores = get_scores(
                df, pretty_print=False, kp_comment_limit=args.kp_comment_cutoff
            )
        else:
            scores = get_scores(df, pretty_print=True)
