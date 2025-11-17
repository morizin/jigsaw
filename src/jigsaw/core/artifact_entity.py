from .io_types import Directory, FilePath
from pydantic import BaseModel
from typing import Optional
from .util_entity import DataSchema
from .util_entity import ClassificationMetric


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
    train_file_path: FilePath
    valid_file_path: Optional[FilePath] = None
    test_file_path: Optional[FilePath] = None


class ModelInferenceArtifact(BaseModel):
    name: str
    prediction_file_path: FilePath


class MultiModelInferenceArtifact(BaseModel):
    outdir: Directory
    models: dict[str, ModelInferenceArtifact]


class ModelTrainingArtifact(BaseModel):
    name: str
    experiment_name: Optional[str] = None
    config_path: Optional[str] = None
    model_path: FilePath | Directory
    metrics: ClassificationMetric


class MultiModelTrainingArtifact(BaseModel):
    outdir: Directory
    models: dict[str, ModelTrainingArtifact]
