from pydantic import BaseModel
from .common import Directory, FilePath


class DataSource(BaseModel):
    source: str
    type: str
    name: str


class DataIngestionConfig(BaseModel):
    sources: list[DataSource]
    names: list[str]
    outdir: Directory


class DataSchema(BaseModel):
    name: str
    schema: dict[str, str]
    train: list[FilePath]
    test: list[FilePath]
    features: list[str]
    target: str


class DataValidationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    statistics: bool
    schemas: list[DataSchema]


class DataSplitParams(BaseModel):
    type: str
    nsplits: int = 5
    random_state: int = 2025
    labels: list[str] | str | None = None


class DataTransformationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    datasets: list[str]
    splitter: DataSplitParams
    features: dict[str, list[str] | None]
    targets: dict[str, str | list[str] | None]
    wash: bool
    triplet: bool
    zero: bool
    pairwise: bool
