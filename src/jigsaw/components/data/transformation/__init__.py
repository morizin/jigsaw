from .... import logger
from ....core import Directory
from ....core.config_entity import DataTransformationConfig
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

        length = min(LENGTH, max(list(map(len, self.schemas.keys())))) + 15
        print("=" * length)
        print("|", "Datasets Available".center(length - 4), "|")
        print("=" * length)
        for name in self.schemas:
            print("|", name.center(length - 4), "|")
        print("=" * length)

        self.pipelines = [PIPELINES["cleaned"]]
        final_dir = []
        for name, process in PIPELINES.items():
            if getattr(self.config, name, False):
                self.pipelines.append(process)
                final_dir.append(name)

        self.final_dir = "_".join(final_dir)
        print(self.final_dir)
        length = (
            min(LENGTH, max(list(map(len, self.schemas.keys()))))
            + 15
            + len(self.final_dir)
        )
        print("=" * length)
        print("|", "Datasets Generating".center(length - 4), "|")
        print("=" * length)
        for name in self.schemas:
            print("|", f"{self.final_dir}_{name}".center(length - 4), "|")
        print("=" * length)

    def __call__(self):
        train_data = []
        test_data = []
        for name, schema in self.schemas.items():
            indir = self.config.indir / name
            for path in schema.train:
                data = load_csv(indir / path)
                for process in self.pipelines:
                    data = process(
                        config=self.config,
                        data=data,
                        dataname=name,
                        filename=path,
                    )
                save_csv(data, self.config.outdir // f"{self.final_dir}_{name}" / path)
                train_data.append(data)

            for path in schema.test:
                data = load_csv(indir / path)
                for process in self.pipelines:
                    data = process(
                        config=self.config,
                        data=data,
                        dataname=name,
                        filename=path,
                    )
                save_csv(data, self.config.outdir // f"{self.final_dir}_{name}" / path)
                test_data.append(data)

        if train_data:
            train_data = pd.concat(train_data, axis=0).reset_index(drop=True)
            save_csv(
                train_data,
                self.config.outdir // f"{self.final_dir}_combined" / "train.csv",
            )

        if test_data:
            test_data = pd.concat(test_data, axis=0).reset_index(drop=True)
            save_csv(
                test_data,
                self.config.outdir // f"{self.final_dir}_combined" / "test.csv",
            )

        return DataTransformationArtifact(
            combined_train_file=self.config.outdir
            / f"{self.final_dir}_combined"
            / "train.csv",
            combined_test_file=self.config.outdir
            / f"{self.final_dir}_combined"
            / "test.csv",
            transformed_outdir=self.config.outdir,
        )
