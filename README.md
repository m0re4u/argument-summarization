
# argument-summarization

Code accompanying the EACL2024 long paper ["An Empirical Analysis of Diversity in Argument Summarization."](https://aclanthology.org/2024.eacl-long.123/)

```
@inproceedings{vandermeer2024empirical,
    title = "An Empirical Analysis of Diversity in Argument Summarization",
    author = "Van Der Meer, Michiel and Vossen, Piek  and Jonker, Catholijn and Murukannaiah, Pradeep",
    editor = "Graham, Yvette and Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.123",
    pages = "2028--2045",
}

```


## Requirements & Setup
See `pyproject.toml`

## Data
Get the data associated with each dataset used in the paper, and put it under the `data/` folder.
- **ArgKP** Download the data [here](https://github.com/IBM/KPA_2021_shared_task/tree/main/kpm_data) and [here](https://github.com/IBM/KPA_2021_shared_task/tree/main/test_data) and extract to `data/kpa2021/`
- **HyEnA** Download the data [here](https://osf.io/4rs5v/) and move the `external_hyena_dataset_new.csv` file into `data/hyena/`
- **Perspectrum** Download the data [here](https://github.com/CogComp/perspectrum/tree/master/data/dataset) and extract to `data/perspectrum/`


## KPA approaches
- **ChatGPT** Make sure you put your OpenAI API Token in a `.env` file (variable name `OPENAI_API_KEY`).
- **Debater** Make sure you put your IBM Debater API Token in a `.env` file (variable name `IBM_API_KEY`).
- **SMatchToPR** We copied the implementation from [here](https://github.com/webis-de/argmining-21-keypoint-analysis-sharedtask-code). Use the original code base to train using contrastive learning, and then use the trained model for running the KPA pipeline. The Perspectrum dataset class can be converted for contrastive training using `PerspectrumDataset.extract_csv_for_smatch()`. See the example script in `test/perspectrum_extract.py`.

## Running KPG
1. `python3 scripts/run_kpa.py`
2. `python3 scripts/evaluate_kpg.py`

## Running KPM
1. `python3 scripts/evaluate_kpm.py --gather_new`

## Analyses
See the notebooks and scripts in `scripts/`.
- `lower_limit_kpg.sh`/`lower_limit.sh`/`perspectrum_sources.sh` experiment files for generating results.
- `table_data_overview.py` for (partially) creating the dataset overview table.