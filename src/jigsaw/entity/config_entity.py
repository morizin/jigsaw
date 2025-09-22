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
    target: str


class DataValidationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    statistics: bool
    schemas: list[DataSchema]
