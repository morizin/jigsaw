from pydantic import BaseModel
from .io_types import Directory
from .entity import DataSchema, DataSource


class DataIngestionConfig(BaseModel):
    sources: dict[str, DataSource]
    outdir: Directory


class DataValidationConfig(BaseModel):
    outdir: Directory
    report_name: str
    indir: Directory
    statistics: bool
    data_drift: bool
    schemas: dict[str, DataSchema]


class DataSplitConfig(BaseModel):
    type: str
    nsplits: int = 5
    random_state: int = 2025
    labels: list[str] | str | None = None


class TripletDataConfig(BaseModel):
    ntriplets: int
    nsamples: int
    random_state: int


class DataTransformationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    datasets: list[str]
    splitter: None | DataSplitConfig
    features: dict[str, list[str] | None]
    targets: dict[str, str | list[str] | None]
    urlparse: bool
    wash: bool
    zero: bool
    triplet: TripletDataConfig | None
    pairwise: bool
    final_dir: str | None = None


class TokenizerConfig(BaseModel):
    max_length: int
    truncation: bool | str
    padding: str | bool


class EngineConfig(BaseModel):
    model_name: str
    nepochs: int
    learning_rate: float
    train_batch_size: int
    valid_batch_size: None | int = None
    gradient_accumulation_steps: int
    weight_decay: float | None
    warmup_ratio: float | None
    tokenizer: TokenizerConfig


class ModelTrainingConfig(BaseModel):
    outdir: Directory
    indir: Directory
    schemas: list[DataSchema]
    fold: int
    engine: EngineConfig
