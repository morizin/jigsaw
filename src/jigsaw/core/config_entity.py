from pydantic import BaseModel, field_validator
from jigsaw.core import FilePath, Directory
from jigsaw import log_dir as LOG_DIR
from .util_entity import (
    DataSchema,
    DataSource,
    DataDriftConfig,
    TripletDataConfig,
    DataSplitConfig,
)
from jigsaw.components.data.augmentation import Augmentor
import json
import os

from typing import Literal, Optional


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
    name: str
    outdir: Directory | FilePath
    type: Literal["text-classification", "triplet", "completion"]
    model: FilePath
    augmentations: Optional[bool | Augmentor] = False
    dataloader_pin_memory: bool = False
    seed: int
    optimizer: str
    max_grad_norm: float

    n_epochs: Optional[int | float] = None
    learning_rate: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    gradient_checkpointing: bool = True

    train_file_path: FilePath
    train_batch_size: Optional[int] = None

    do_eval: bool = False
    valid_file_path: Optional[FilePath] = None
    valid_batch_size: Optional[int] = None
    eval_strategy: str = "no"
    eval_steps: Optional[int | float] = None
    eval_epochs: Optional[int | float] = None

    logging_strategy: str = "no"
    logging_dir: Optional[Directory | FilePath] = None
    logging_steps: Optional[int | float] = None
    logging_epochs: Optional[int | float] = None

    save_strategy: str = "no"
    save_steps: Optional[int | float] = None
    save_epochs: Optional[int | float] = None

    scheduler_type: Literal["cosine", "linear"] = "cosine"
    warmup_ratio: Optional[float] = None

    max_length: Optional[int] = None
    padding: Optional[bool | str] = None
    fold: int = -1
    out_of_fold: bool = False

    model_config = {
        "ser_json_t": True,
        "json_encoders": {
            Directory: lambda v: str(v.path),
            FilePath: lambda v: str(v),
        },
    }

    @field_validator("outdir", mode="before")
    @classmethod
    def fix_outdir(cls, outdir: FilePath) -> Directory:
        if isinstance(outdir, FilePath):
            outdir = Directory(path=outdir)
        return outdir

    def model_post_init(self, __context):
        if os.path.basename(self.outdir.path) != self.name:
            self.outdir //= self.name

        self.out_of_fold = self.fold >= 0

        if self.valid_batch_size:
            if self.eval_epochs:
                self.eval_strategy = "epochs"
            elif self.eval_steps:
                self.eval_strategy = "steps"
            else:
                self.eval_strategy = "no"

            if self.eval_strategy != "no":
                self.do_eval = True

        if self.logging_epochs:
            self.logging_strategy = "epochs"
        elif self.logging_steps:
            self.logging_strategy = "steps"

        if self.logging_strategy != "no" and (
            self.logging_dir is None or os.path.basename(self.logging_dir) != self.name
        ):
            self.logging_dir = str(
                Directory(path=self.logging_dir or LOG_DIR) // self.name
            )

    def to_mlflow_params(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ):
        data = self.model_dump(mode="json")

        # Filter fields if include is defined
        if exclude:
            data = {k: v for k, v in data.items() if k not in exclude}

        if include:
            data = {k: v for k, v in data.items() if k in include}

        # Flatten & stringify
        params = {}
        for k, v in data.items():
            # MLflow does not allow dicts, lists, or None
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            if v is None:
                v = "None"
            params[k] = str(v)

        return params


class MultiModelTrainingConfig(BaseModel):
    outdir: Directory
    models: dict[str, FilePath]


class ModelInferenceConfig(BaseModel):
    name: str
    type: str
    outdir: Directory | FilePath
    model_path: Directory | FilePath
    batch_size: Optional[int] = None
    train_path: Optional[Directory | FilePath] = None
    tta: bool | Augmentor = False
    max_length: int = 256
    ensemble_weight: int = 1

    model_config = {
        "ser_json_t": True,
        "json_encoders": {
            Directory: lambda v: str(v.path),
            FilePath: lambda v: str(v),
        },
    }

    @field_validator("outdir", mode="before")
    @classmethod
    def fix_outdir(cls, outdir: FilePath) -> Directory:
        if isinstance(outdir, FilePath):
            outdir = Directory(path=outdir)
        return outdir

    def model_post_init(self, __context):
        if os.path.basename(self.outdir.path) != self.name:
            self.outdir //= self.name


class MultiModelInferenceConfig(BaseModel):
    outdir: Directory
    models: dict[str, FilePath]
