from pydantic import BaseModel, model_validator
from .io_types import Directory, FilePath
from ..constants import SCHEMA_DIR
from .. import logger
from glob import glob
import os
import yaml


class DataSchema(BaseModel):
    name: str
    path: FilePath | Directory

    train: list[FilePath] | None = None
    test: list[FilePath] | None = None
    submission: list[FilePath] | None = None

    schema: dict[str, str] | None = None
    categorical: list[str] | None = None
    filepath: list[str] | None = None
    features: list[str] | None = None
    target: str | list[str] | None = None

    @model_validator(mode="after")
    def get_schema(self):
        file_path = os.path.join(SCHEMA_DIR, f"{self.name}.yaml")
        if os.path.exists(file_path):
            content = yaml.safe_load(open(file_path, "r"))
            self.schema = content["columns"]
            self.features = content["features"]
            self.categorical = content.get("categorical", [])
            self.filepath = (
                content.get("filepath") if content.get("filepath", None) else []
            )
            self.target = content["target"]
            return self
        else:
            e = f"Schema of dataset '{self.name}' not found"
            logger.error(e)
            raise Exception(e)

    @model_validator(mode="after")
    def get_files(self):
        files = glob(f"{self.path}/**/*", recursive=True)
        files = [file[len(str(self.path)) + 1 :] for file in files]
        self.test = list(filter(lambda x: "test" in x, files))
        self.submission = list(filter(lambda x: "sub" in x, files))
        self.train = list(filter(lambda x: "test" not in x and "sub" not in x, files))
        return self


class DataSource(BaseModel):
    source: str
    type: str
    uri: str


class DataDriftConfig(BaseModel):
    n_splits: int = 5
    n_iterations: int = 100
    dimension: int = 500


class DataSplitConfig(BaseModel):
    type: str
    n_splits: int = 5
    labels: list[str] | str | None


class TripletDataConfig(BaseModel):
    anchor_col: str
    sample_col: str
    n_negatives: int = 5
    n_samples: int = 5
    reversed: bool = False


class ClassificationMetric(BaseModel):
    roc_auc: float
    accuracy: float
