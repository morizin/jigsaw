from pydantic import validate_call
import pandas as pd
from src.jigsaw import logger
from src.jigsaw.entity.common import FilePath
from pathlib import Path
from src.jigsaw.entity.config_entity import DataValidationConfig, DataSchema
from src.jigsaw.utils.common import read_csv, save_json
from collections import defaultdict


class DataValidationComponent:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.names = [i.name for i in self.config.schemas]
        tree = lambda: defaultdict(tree)
        self.status = tree()

        self.outdir = self.config.outdir.path
        self.indir = self.config.indir.path
        length = 50
        print("=" * length)
        string = "Datasets Available"
        space = length - len(string) - 2
        print(
            "|{}{}{}|".format(
                " " * (space // 2),
                string,
                " " * (space // 2),
            )
        )

        print("=" * length)
        for name in self.names:
            space = length - len(name) - 1
            print(
                "|{}{}{}|".format(
                    " " * (space // 2),
                    name,
                    " " * (space // 2),
                )
            )
        print("=" * length)

    def validate_all(self):
        for schema in self.config.schemas:
            for file in schema.train:
                self.find_missing_columns(
                    self.indir / schema.name / file, schema, train=True
                )

            for file in schema.test:
                self.find_missing_columns(
                    self.indir / schema.name / file, schema, train=False
                )

        save_json(self.status, self.outdir / "status.json")

    @validate_call
    def find_missing_columns(self, data_path: Path, schema: DataSchema, train=True):
        data = read_csv(data_path)
        data_cols = data.columns
        if not train:
            schema.schema.pop(schema.target)
        target_cols = schema.schema.keys()

        validation_status = True
        for col in target_cols:
            if col not in data_cols:
                logger.error(f"Missing column {col} in file {data_path.as_posix()}")
                validation_status = False
            else:
                validation_status &= True

        self.status[str(data_path)]["missing_values"] = not validation_status
        if validation_status:
            logger.info(f"{data_path} : No Missing Values")

    def get_statistics(self, path: FilePath):
        pass
