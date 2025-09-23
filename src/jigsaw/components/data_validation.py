from pydantic import validate_call
from src.jigsaw import logger
from src.jigsaw.entity.common import FilePath
from pathlib import Path
from src.jigsaw.entity.config_entity import DataValidationConfig, DataSchema
from src.jigsaw.utils.common import read_csv, save_json
from collections import defaultdict
from pandas.api.types import is_object_dtype, is_integer_dtype

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
                self.find_mismatch_dtype(
                    self.indir / schema.name / file, schema, train=True
                )

            for file in schema.test:
                self.find_missing_columns(
                    self.indir / schema.name / file, schema, train=False
                )
                self.find_mismatch_dtype(
                    self.indir / schema.name / file, schema, train=False
                )

        save_json(self.status, self.outdir / "status.json")

    @validate_call
    def find_missing_columns(self, data_path: FilePath, schema: DataSchema, train=True):
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

        self.status[f"{schema.name}.{data_path.name}"]["missing_values"] = not validation_status
        if validation_status:
            logger.info(f"[SUCCESS] Check Missing Columns: {schema.name}.{data_path.name}")
        else:
            logger.error(f"[FAILED] Check Missing Columns: {schema.name}.{data_path.name}")
            raise Exception(f"File {data_path.name} of Dataset {schema.name} have missing values")

    @validate_call
    def find_mismatch_dtype(self, data_path: FilePath, schema: DataSchema, train = True):
        validation_status = True
        data = read_csv(data_path)

        for (column, dtype) in schema.schema.items():
            if dtype.lower() in ('str', 'string', 'object'):
                if not is_object_dtype(data[column]):
                    logger.error("Mismatch DataType : {schema.name}.{data_path.name}.{column}.dtype != {dtype}")
                    validation_status = False
                else:
                    validation_status &= True

            if dtype.lower() in ('int', 'int32', 'int64', "integer","natural"):
                if not is_integer_dtype(data[column]):
                    validation_status = False
                    logger.error("Mismatch DataType : {schema.name}.{data_path.name}.{column}.dtype != {dtype}") 
                else:
                    validation_status &= True

        self.status[f"{schema.name}.{data_path.name}"]['mismatch_dtype'] = not validation_status

        if validation_status:
            logger.info(f"[SUCCESS] Check Mismatch Datatype: {schema.name}.{data_path.name}")
        else:
            logger.error(f"[FAILED] Check Mismatch Datatype: {schema.name}.{data_path.name}")

    def get_statistics(self, path: FilePath):
        pass
