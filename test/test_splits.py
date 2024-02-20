from argument_summarization.data import HyEnADataset, KPA2021Dataset, PerspectrumDataset

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
        PerspectrumDataset("data/perspectrum/", split="test", limit_topics_frac=0.2),
        PerspectrumDataset("data/perspectrum/", split="test", limit_source="idebate"),
        PerspectrumDataset("data/perspectrum/", split="test", limit_source="procon"),
        PerspectrumDataset(
            "data/perspectrum/", split="test", limit_source="debatewise"
        ),
    ]

    for dataset in datasets:
        print(
            f"dataset: {dataset.dataset_name} {dataset.split} - t: {len(dataset.get_topics())} c: {len(dataset.get_comments())} "
        )
        print(len(dataset.kpm_data))
