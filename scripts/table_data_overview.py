from argument_summarization.data import HyEnADataset, KPA2021Dataset, PerspectrumDataset
import numpy as np

if __name__ == "__main__":
    # Load data
    datasets = [
        KPA2021Dataset("data/kpa2021/", split="train"),
        KPA2021Dataset("data/kpa2021/", split="val"),
        KPA2021Dataset("data/kpa2021/", split="test"),
        HyEnADataset("data/hyena/", split="test"),
        PerspectrumDataset("data/perspectrum/", split="train"),
        PerspectrumDataset("data/perspectrum/", split="val"),
        PerspectrumDataset("data/perspectrum/", split="test"),
    ]
    # For the table in the main paper
    for dataset in datasets:
        print(
            f"dataset: {dataset.dataset_name} {dataset.split} - t: {len(dataset.get_topics())} c: {len(dataset.get_comments())} "
        )

    # Appendix table detailed
    for dataset in datasets:
        x = [len(c) for c in dataset.get_comments()]
        comments = [x[:2999] if len(x) >= 3000 else x for x in dataset.get_comments()]
        x2 = [len(c) for c in comments]
        print(
            f"dataset: {dataset.dataset_name} {dataset.split} - {np.mean(x):.0f} - {np.mean(x2):.0f}"
        )
