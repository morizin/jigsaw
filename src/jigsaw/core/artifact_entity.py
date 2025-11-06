from .io_types import Directory, FilePath
from pydantic import BaseModel
from .entity import DataSchema


class DataIngestionArtifact(BaseModel):
    path: Directory | FilePath
    names: list[str]


class DataValidationArtifact(BaseModel):
    valid_outdir: Directory | FilePath
    invalid_outdir: Directory | FilePath
    report_dir: Directory | FilePath
    schemas: dict[str, DataSchema]


class DataTransformationArtifact(BaseModel):
    valid_outdir: Directory | FilePath
    invalid_outdir: Directory | FilePath
    report_dir: Directory | FilePath
    schemas: dict[str, DataSchema]
