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

class TripletDataConfig(BaseModel):
    ntriplets : int
    nsamples  : int
    random_state : int

class DataTransformationConfig(BaseModel):
    outdir: Directory
    indir: Directory
    datasets: list[str]
    splitter: DataSplitParams
    features: dict[str, list[str] | None]
    targets: dict[str, str | list[str] | None]
    urlparse : bool
    wash: bool
    zero: bool
    triplet: TripletDataConfig | None 
    pairwise: bool
    final_dir: str | None = None

class TokenizerParams(BaseModel):
    max_length : int
    truncation : bool | str
    padding : str | bool

class EngineParams(BaseModel):
    model_name : str
    nepochs : int
    learning_rate : float
    batch_size : int
    gradient_accumulation_step : int
    weight_decay : float | None
    warmup_ratio : float | None
    tokenizer : TokenizerParams

class ModelTrainingConfig(BaseModel):
    outdir : Directory
    datasets: list[str] | str
    fold : int
    engine : EngineParams

