from argument_summarization.data import PerspectrumDataset

if __name__ == "__main__":
    # Load data
    datasets = [
        PerspectrumDataset("data/perspectrum/", split="train"),
        PerspectrumDataset("data/perspectrum/", split="val"),
    ]

    for dataset in datasets:
        dataset.extract_csv_for_smatch()
