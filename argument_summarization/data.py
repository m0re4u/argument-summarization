import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Union
from itertools import product

import pandas as pd
import numpy as np


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


class KPADataset:
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    data: pd.DataFrame
    stance_mapping: Dict[str, int]
    kpm_data: pd.DataFrame

    def get_comments(
        self, with_stance=None, with_topic=None, all_data=False
    ) -> Union[List[str], pd.DataFrame]:
        data = self.data
        if with_stance is not None:
            data = data[data["stance"] == self.stance_mapping[with_stance]]

        if with_topic is not None:
            data = data[data["topic"] == with_topic]

        if all_data:
            return data
        else:
            return data["argument"].tolist()

    def get_key_points(self) -> List[str]:
        return self.data["key_point"].unique().tolist()

    def get_topics(self) -> List[str]:
        return self.data["topic"].unique().tolist()

    def get_kpm_data(self) -> pd.DataFrame:
        return self.kpm_data

    def check_mapping(self) -> bool:
        """
        Answers the question:
            Does this dataset have a many to one mapping from arguments to key points?
        """
        arg_to_kp = defaultdict(list)
        for _, row in self.kpm_data.iterrows():
            if row["label"] == 1:
                arg_to_kp[row["argument"]].append(row["key_point"])
        x = len([k for k, v in arg_to_kp.items() if len(v) > 1]) > 0
        return x


class KPA2021Dataset(KPADataset):
    dataset_name: str = "kpa2021"

    def __init__(self, path: str, split=None):
        self.dataset_dir = Path(path)
        self.load_shared_task_data()
        # Merge into one big dataset
        self.merge_data()

        self.stance_mapping = {
            "pro": 1,
            "con": -1,
        }

        # Get KPM data
        self.kpm_data = self.create_kpm_data(
            self.labels,
            self.raw_data,
            self.key_points,
        )

        self.split = split
        if split is not None:
            self.data = self.data[self.data.split == split]
            self.kpm_data = self.kpm_data[self.kpm_data.split == split]

    def load_shared_task_data(self):
        """
        Load the KPA2021 data from files and remap the arg_id and key_point_id counters in the test set.
        """
        self.train_data = pd.read_csv(self.dataset_dir / "arguments_train.csv")
        self.val_data = pd.read_csv(self.dataset_dir / "arguments_dev.csv")
        self.test_data = pd.read_csv(self.dataset_dir / "arguments_test.csv")
        self.labels_train = pd.read_csv(self.dataset_dir / "labels_train.csv")
        self.labels_val = pd.read_csv(self.dataset_dir / "labels_dev.csv")
        self.labels_test = pd.read_csv(self.dataset_dir / "labels_test.csv")
        self.key_points_train = pd.read_csv(self.dataset_dir / "key_points_train.csv")
        self.key_points_val = pd.read_csv(self.dataset_dir / "key_points_dev.csv")
        self.key_points_test = pd.read_csv(self.dataset_dir / "key_points_test.csv")
        # name the splits
        self.train_data["split"] = "train"
        self.val_data["split"] = "val"
        self.test_data["split"] = "test"

        # Offset arg and kp counters since they are overlapping for the test data
        self.test_data["arg_id"] = self.test_data["arg_id"].apply(
            lambda x: self.offset_id_counter(x)
        )
        self.key_points_test["key_point_id"] = self.key_points_test[
            "key_point_id"
        ].apply(lambda x: self.offset_id_counter(x))
        self.labels_test["key_point_id"] = self.labels_test["key_point_id"].apply(
            lambda x: self.offset_id_counter(x)
        )
        self.labels_test["arg_id"] = self.labels_test["arg_id"].apply(
            lambda x: self.offset_id_counter(x)
        )

    def merge_data(self, keep_only_rematched=False):
        self.raw_data = pd.concat([self.train_data, self.val_data, self.test_data])
        self.key_points = pd.concat(
            [self.key_points_train, self.key_points_val, self.key_points_test]
        )
        self.labels = pd.concat([self.labels_train, self.labels_val, self.labels_test])
        labels = self.labels

        if keep_only_rematched:
            # We can consider only keeping the matched argument and key point pairs
            labels = self.labels[self.labels["label"] == 1].reset_index(drop=True)

        # merge arguments and labels
        self.data = pd.merge(
            self.raw_data,
            labels[["arg_id", "key_point_id"]],
            on="arg_id",
            how="left",
        )
        # merge arguments and key points
        self.data = pd.merge(
            self.data,
            self.key_points[["key_point_id", "key_point"]],
            on="key_point_id",
            how="left",
        )

    def create_kpm_data(self, labels, argument_data, key_point_data):
        """
        Create a separate dataframe for KPM data since we also want negative labels (not matching arguments and key points )
        """
        kpm_data = pd.merge(labels, argument_data, on="arg_id", how="left")
        kpm_data.dropna(inplace=True)
        kpm_data = pd.merge(
            kpm_data,
            key_point_data[["key_point_id", "key_point"]],
            on="key_point_id",
            how="left",
        )
        kpm_data.dropna(inplace=True)
        return kpm_data

    def offset_id_counter(self, item, offset=28, id_index=1):
        """
        Offset the arg_id and key_point_id counters.
        """
        seps = item.split("_")
        seps[id_index] = str(int(seps[id_index]) + offset)
        return "_".join(seps)


class HyEnADataset(KPADataset):
    dataset_name: str = "hyena"

    def __init__(self, path: str, split="test"):
        self.topics = {
            "1": "Young people may come together in small groups",
            "2": "All restrictions are lifted for persons who are immune",
            "3": "Reopen hospitality and entertainment industry",
        }
        self.stance_mapping = {
            "pro": "pro",
            "con": "con",
        }
        self.dataset_path = Path(path) / "external_hyena_dataset_new.csv"
        self.data = pd.read_csv(self.dataset_path)
        # rename columns
        self.data = self.data.rename(
            {"motivation": "argument", "key_argument": "key_point"}, axis=1
        )
        # Map topic
        self.split = split
        if split != "test":
            raise NotImplementedError("Only test split is supported for HyEnA")
        self.data["topic"] = self.data["topic"].apply(lambda x: self.topics[str(x)])
        self.kpm_data = self.get_negatives()

        # load the key points as reported in the plos one paper analzying the comments manually
        self.manual_key_points = {
            "1": [
                "Young people play a minor role in the spread of the virus and their risk of getting sick is low",
                "Social contact is relatively important for young people (to develop themselves)",
                "For young people it is difficult not to violate the rules",
                "Reduction of problematic psychological symptoms",
                "Reduces the pressure on parents",
                "Possibility to build up herd immunity",
                "Increases support among young people for other lockdown measures",
                "Constitutes age discrimination which results in a dichotomy in society",
                "Measures are difficult to enforce. Young people will also get in contact with other people",
            ],
            "2": [
                "These people pose no danger to their environment",
                "These people can keep society and the economy going again",
                "It is pointless to demand solidarity from these people if they are already immune. Doing so will lead to fierce protests",
                "Tests for immunity are not foolproof, and this increases the risk of new infections",
                "Creates a dichotomy in society. People who are not immune can get annoyed by the behaviour of those who are allowed to resume normal life",
                "Difficult to enforce",
                "Potential confusion as immunity is not outwardly apparent",
            ],
            "3": [
                "This is good for our economy and business",
                "It is good for people’s well-being",
                "This relaxation option will increase support for the continuation of the other measures",
                "It is enforceable",
                "People can take responsibility for themselves by staying away if they wish",
                "We should preserve our cultural heritage and cannot risk bankruptcies in the cultural sector",
                "Keeping these businesses closed is too big of a sacrifice for young people",
                "In this way, we can build up herd immunity",
                "If the hospitality industry is not re-opened people will do other things to relax which is also risky",
                "Risk of too many people gathering together, which helps to spread the virus",
                "It is not necessary at the moment",
                "When alcohol is consumed, people are more likely to underestimate risks and are less likely to comply with distancing measures",
                "Opening up the hospitality and entertainment sectors should only be considered in the next phase if it appears that other adjustments have worked",
                "Hospitality industry has a bad impact on society. Please keep it closed",
            ],
        }

    def get_negatives(self, upsample: float = 4):
        """
        Get negative labels for KPM data by shuffling arguments and key points.
        """
        df_copy = self.data.copy()
        df_copy["label"] = 1
        new_data = []
        for topic, topic_df in df_copy.groupby("topic"):
            upsample_len = int(len(topic_df) * upsample)

            # take existing argument/key point combinations
            arg_kp_combinations = [
                (x, y) for x, y in topic_df[["argument", "key_point"]].values
            ]
            arg_kp_combinations = set(arg_kp_combinations)
            # create all possible combinations
            args = set(topic_df["argument"].astype(str))
            kps = set(topic_df["key_point"].astype(str))

            all_combinations = set(product(args, kps))
            # remove existing combinations
            negative_combinations = all_combinations - arg_kp_combinations
            # create dataframe
            negative_df = pd.DataFrame(
                list(negative_combinations), columns=["argument", "key_point"]
            )
            negative_df["topic"] = topic
            negative_df["label"] = 0
            new_data.append(negative_df.sample(n=upsample_len, replace=False))
            new_data.append(topic_df.drop(["stance"], axis=1))

        return pd.concat(new_data)


class ThinkPDataset(KPADataset):
    dataset_name: str = "thinkp"

    def __init__(self, path: str):
        self.dataset_path = Path(path) / "ThinkP.jsonl"

        self.data = pd.read_json(path_or_buf=self.dataset_path, lines=True)
        raise NotImplementedError("ThinkP dataset is not implemented yet")


class PerspectrumDataset(KPADataset):
    dataset_name: str = "perspectrum"

    def __init__(self, path: str, split=None, limit_topics_frac=1.0, limit_source=None):
        self.dataset_dir = Path(path)
        self.load_from_file()
        self.data["pId"] = self.data["pId"].apply(lambda x: x[0])
        self.data = pd.merge(
            self.data,
            self.evidence[["eId", "text"]],
            on="eId",
            how="left",
        )
        self.data = pd.merge(
            self.data,
            self.perspective[["pId", "text"]],
            on="pId",
            how="left",
            suffixes=("_argument", "_key_point"),
        )
        self.data = self.data.rename(
            {
                "text_argument": "argument",
                "text_key_point": "key_point",
                "stance_label_3": "stance",
            },
            axis=1,
        )
        self.data.dropna(inplace=True)
        self.data["split"] = self.data["cId"].apply(lambda x: self.split_data[x])
        if limit_source is not None:
            if limit_source not in set(self.data.source):
                raise ValueError(f"Source {limit_source} not in dataset")
            self.data = self.data[self.data.source == limit_source]
        self.stance_mapping = {"con": "UNDERMINE", "pro": "SUPPORT"}

        self.kpm_data = self.get_negatives()

        self.split = split
        if split is not None:
            self.data = self.data[self.data.split == split]
            self.kpm_data = self.kpm_data[self.kpm_data.split == split]

        if limit_topics_frac < 1.0:
            X = int(len(self.data.topic.unique()) * limit_topics_frac)
            np.random.seed(42)
            print(f"Limiting to {X} topics")
            selected_topics = np.random.choice(
                self.data.topic.unique(), X, replace=False
            )
            self.data = self.data[self.data.topic.isin(selected_topics)]
            self.kpm_data = self.kpm_data[self.kpm_data.topic.isin(selected_topics)]

    def load_from_file(self):
        self.mappig_path = self.dataset_dir / "perspectrum_with_answers_v1.0.json"
        self.perspective_path = self.dataset_dir / "perspective_pool_v1.0.json"
        self.evidence_path = self.dataset_dir / "evidence_pool_v1.0.json"
        self.split_path = self.dataset_dir / "dataset_split_v1.0.json"
        df = pd.read_json(path_or_buf=self.mappig_path, lines=False)
        self.evidence = pd.read_json(path_or_buf=self.evidence_path, lines=False)
        self.perspective = pd.read_json(path_or_buf=self.perspective_path, lines=False)
        self.split_data = pd.read_json(
            path_or_buf=self.split_path, lines=False, typ="series"
        )
        self.split_data = self.split_data.map(
            {"train": "train", "dev": "val", "test": "test"}
        )
        # Claims are topics
        # Perspectives are key points
        # Evidence are arguments
        self.data = df.rename({"text": "topic", "topics": "keywords"}, axis=1)
        self.data = self.data.explode("perspectives").reset_index(drop=True)
        df2 = pd.json_normalize(self.data.perspectives)
        self.data = pd.concat([self.data, df2], axis=1)
        self.data = self.data.explode("evidence").reset_index(drop=True)
        self.data = self.data.rename({"evidence": "eId", "pids": "pId"}, axis=1)

    def get_negatives(self, upsample: float = 4):
        """
        Get negative labels for KPM data by shuffling arguments and key points.
        """
        df_copy = self.data.copy()
        df_copy["label"] = 1
        new_data = []
        for topic, topic_df in df_copy.groupby("topic"):
            split = topic_df["split"].iloc[0]
            upsample_len = int(len(topic_df) * upsample)

            # take existing argument/key point combinations
            arg_kp_combinations = [
                (x, y) for x, y in topic_df[["argument", "key_point"]].values
            ]
            arg_kp_combinations = set(arg_kp_combinations)
            # create all possible combinations
            args = set(topic_df["argument"].astype(str))
            kps = set(topic_df["key_point"].astype(str))

            all_combinations = set(product(args, kps))
            # remove existing combinations
            negative_combinations = all_combinations - arg_kp_combinations
            # create dataframe
            negative_df = pd.DataFrame(
                list(negative_combinations), columns=["argument", "key_point"]
            )
            negative_df["topic"] = topic
            negative_df["label"] = 0

            try:
                new_samples = negative_df.sample(n=upsample_len, replace=False)
            except ValueError:
                # We do not have enough samples to upsample to the desired length
                # Simply take the max number of possible negative samples
                upsample_len = len(negative_df)
                new_samples = negative_df.sample(n=upsample_len, replace=True)

            new_samples["split"] = split
            new_data.append(new_samples)
            new_data.append(topic_df.drop(["stance"], axis=1))

        return pd.concat(new_data)

    def extract_csv_for_smatch(self):
        extract_df = self.kpm_data[["topic", "argument", "key_point", "label"]].copy()
        extract_df["ex_keypoint"] = extract_df.apply(
            lambda x: x["topic"] + " <SEP> " + x["key_point"], axis=1
        )
        extract_df["ex_label"] = extract_df.label.apply(lambda x: int(x))
        if self.split is None:
            extract_df[["argument", "ex_keypoint", "ex_label"]].to_csv(
                self.dataset_dir / f"perspectrum_df_contrastive.csv"
            )
        else:
            extract_df[["argument", "ex_keypoint", "ex_label"]].to_csv(
                self.dataset_dir / f"perspectrum_df_contrastive_{self.split}.csv"
            )
