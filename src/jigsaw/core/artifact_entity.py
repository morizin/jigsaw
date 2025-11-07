from .io_types import Directory, FilePath
from pydantic import BaseModel
from .util_entity import DataSchema


class DataIngestionArtifact(BaseModel):
    path: Directory | FilePath
    names: list[str]


class DataValidationArtifact(BaseModel):
    validation_status: bool
    valid_outdir: Directory | FilePath
    invalid_outdir: Directory | FilePath
    report_dir: Directory | FilePath
    schemas: dict[str, DataSchema]


class DataTransformationArtifact(BaseModel):
    transformed_outdir: Directory | FilePath
    combined_train_file: FilePath
    combined_test_file: FilePath
