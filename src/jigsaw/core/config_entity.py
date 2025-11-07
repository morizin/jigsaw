from pydantic import BaseModel
from .io_types import Directory
from .util_entity import (
    DataSchema,
    DataSource,
    DataDriftConfig,
    TripletDataConfig,
    EngineConfig,
    DataSplitConfig,
)


class DataIngestionConfig(BaseModel):
    sources: dict[str, DataSource]
    outdir: Directory


class DataValidationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    report_name: str
    statistics: bool
    data_drift: DataDriftConfig | None
    schemas: dict[str, DataSchema]


class DataTransformationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    schemas: dict[str, DataSchema]
    zero_shot: bool
    pairwise: bool
    splitter: DataSplitConfig | bool
    triplet: TripletDataConfig | bool
    cache_intermediate: bool = False


class ModelTrainingConfig(BaseModel):
    outdir: Directory
    indir: Directory
    schemas: list[DataSchema]
    fold: int
    engine: EngineConfig
