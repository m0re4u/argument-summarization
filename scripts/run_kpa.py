import argparse

from argument_summarization.data import HyEnADataset, KPA2021Dataset, PerspectrumDataset
from argument_summarization.kpa import KPS_APIKPA, ChatGPTKPA, SMatchToPRKPA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, help="Output directory", default="results"
    )
    args = parser.parse_args()

    # Load data
    datasets = [
        KPA2021Dataset("data/kpa2021/", split="test"),
        HyEnADataset("data/hyena/", split="test"),
        PerspectrumDataset("data/perspectrum/", split="test"),
    ]

    # Load methods
    methods = [
        ChatGPTKPA(outdir=args.outdir),
        ChatGPTKPA(outdir=args.outdir, closed_book=False),
        KPS_APIKPA(outdir=args.outdir),
        SMatchToPRKPA(
            "models/roberta-large-contrastive",
            outdir=args.outdir,
            create_new_kps=True,
        ),
    ]

    for dataset in datasets:
        for method in methods:
            all_results = method(dataset)

            if all_results is None:
                print(f"No results for {method.name} -> {dataset.dataset_name}")
                continue
