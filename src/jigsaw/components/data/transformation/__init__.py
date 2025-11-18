from ....core import DataTransformationConfig, DataSplitConfig
from ....core.artifact_entity import DataTransformationArtifact

from .cleaning import remove_duplicates
from .zeroshot import zero_shot_transform
from .folding import split_dataset
from .triplet import triplet_dataset
from ....constants import LENGTH
from ....utils.common import load_csv, save_csv
from typeguard import typechecked
import pandas as pd

PIPELINES = {
    "cleaned": remove_duplicates,
    "zero_shot": zero_shot_transform,
    "splitter": split_dataset,
    "triplet": triplet_dataset,
}


class DataTransformationComponent:
    @typechecked
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.schemas = config.schemas

        final_dir = ["cleaned"]
        for name in PIPELINES.keys():
            if getattr(self.config, name, False):
                final_dir.append(name)

        self.final_dir = "_".join(final_dir)

    def __call__(self):
        train_datas = []
        valid_datas = []
        test_datas = []
        train_splits = 10
        while any(
            map(
                lambda x: int(train_splits * x) != train_splits * x,
                self.config.train_splits,
            )
        ):
            train_splits *= 10

        test_splits = 10
        while any(
            map(
                lambda x: int(test_splits * x) != test_splits * x,
                self.config.test_splits,
            )
        ):
            test_splits *= 10

        for name, schema in self.schemas.items():
            indir = self.config.indir / name
            for path in schema.train:
                data = load_csv(indir / path)
                data = remove_duplicates(
                    config=self.config,
                    data=data,
                    dataname=name,
                    filename=path,
                )
                if getattr(self.config, "zero_shot", False):
                    data = zero_shot_transform(
                        config=self.config, data=data, dataname=name, filename=path
                    )

                self.config.splitter = DataSplitConfig(
                    type="mlskfold",
                    n_splits=train_splits,
                    labels=["rule", "rule_violation"],
                )

                data = split_dataset(
                    self.config, data=data, dataname=name, filename=path
                )

                save_csv(data, self.config.outdir // f"{self.final_dir}_{name}" / path)

                train_fold = int(train_splits * self.config.train_splits[0])
                train_data = data.query(f"fold < {train_fold}").drop(["fold"], axis=1)

                valid_folds = list(
                    map(
                        lambda x: x % test_splits,
                        range(
                            train_fold,
                            train_fold
                            + int(train_splits * self.config.train_splits[1]),
                        ),
                    )
                )
                valid_data = data[data["fold"].isin(valid_folds)].drop(["fold"], axis=1)

                if train_data.shape[0]:
                    train_datas.append(train_data)
                if valid_data.shape[0]:
                    valid_datas.append(valid_data)

            for path in schema.test:
                data = load_csv(indir / path)
                data = remove_duplicates(
                    config=self.config,
                    data=data,
                    dataname=name,
                    filename=path,
                )
                if getattr(self.config, "zero_shot", False):
                    data = zero_shot_transform(
                        config=self.config, data=data, dataname=name, filename=path
                    )

                test_data = data[~data[schema.target].isin([0, 1])].reset_index(
                    drop=True
                )
                data = data[data[schema.target].isin([0, 1])].reset_index(drop=True)

                self.config.splitter = DataSplitConfig(
                    type="mlskfold",
                    n_splits=test_splits,
                    labels=["rule", "rule_violation"],
                )

                data = split_dataset(
                    self.config, data=data, dataname=name, filename=path
                )

                save_csv(data, self.config.outdir // f"{self.final_dir}_{name}" / path)

                train_fold = int(test_splits * self.config.test_splits[0])
                train_data = data.query(f"fold < {train_fold}").drop(["fold"], axis=1)

                valid_folds = list(
                    map(
                        lambda x: x % test_splits,
                        range(
                            train_fold,
                            train_fold + int(test_splits * self.config.test_splits[1]),
                        ),
                    )
                )
                valid_data = data[data["fold"].isin(valid_folds)].drop(["fold"], axis=1)

                if train_data.shape[0]:
                    train_datas.append(train_data)
                if valid_data.shape[0]:
                    valid_datas.append(valid_data)
                if test_data.shape[0]:
                    test_datas.append(test_data)

        if train_datas:
            train_data = pd.concat(train_datas, axis=0).reset_index(drop=True)
            save_csv(
                train_data,
                self.config.outdir // f"{self.final_dir}_combined" / "train.csv",
            )

        if valid_datas:
            valid_data = pd.concat(valid_datas, axis=0).reset_index(drop=True)
            save_csv(
                valid_data,
                self.config.outdir // f"{self.final_dir}_combined" / "valid.csv",
            )

        if test_datas:
            test_data = pd.concat(test_datas, axis=0).reset_index(drop=True)
            save_csv(
                test_data,
                self.config.outdir // f"{self.final_dir}_combined" / "test.csv",
            )

        print(train_data.shape, valid_data.shape, test_data.shape)

        return DataTransformationArtifact(
            train_file_path=self.config.outdir
            / f"{self.final_dir}_combined"
            / "train.csv"
            if train_datas
            else None,
            valid_file_path=self.config.outdir
            / f"{self.final_dir}_combined"
            / "valid.csv"
            if valid_datas
            else None,
            test_file_path=self.config.outdir
            / f"{self.final_dir}_combined"
            / "test.csv"
            if test_datas
            else None,
            transformed_outdir=self.config.outdir,
        )
