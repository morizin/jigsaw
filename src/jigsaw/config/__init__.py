import os
from ..utils.common import seed_everything, load_yaml, save_json, get_hw_details
from ..constants import (
    CONFIG_FILE_PATH,
    DATA_DIRECTORY_NAME,
    MODELS_DIRECTORY_NAME,
    SCHEMA_DIR,
)
from ..constants.data import INGESTED_DATA_FOLDER, REPORT_NAME, TRANSFORM_DIR_NAME
from ..errors import DataNotFoundError, DirectoryNotFoundError, ConfigurationError
from ..components.data.augmentation import Augmentor
from typing import Iterator

from ..core import (
    FilePath,
    Directory,
    DataSource,
    DataSchema,
    DataDriftConfig,
    TripletDataConfig,
    DataSplitConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    MultiModelTrainingConfig,
    ModelInferenceConfig,
    MultiModelInferenceConfig,
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    MultiModelTrainingArtifact,
)
from .. import logger, timestamp
from box import ConfigBox
from typeguard import typechecked


class ConfigurationManager:
    @typechecked
    def __init__(
        self,
        config_path: FilePath = CONFIG_FILE_PATH,
    ):
        self.config: ConfigBox = load_yaml(config_path)
        self.seed = self.config.seed
        seed_everything(self.seed)

        n_procs, n_threads = get_hw_details()
        self.config.n_procs = n_procs
        self.config.n_threads = n_threads

        logger.info(f"Running on {n_procs} CPU(s) with {n_threads} Thread(s) each")

        self.artifact_root: Directory = (
            Directory(path=self.config.artifact_root) // timestamp
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        sources = self.config.data.sources

        sources: dict[str, DataSource] = {
            name: DataSource(source=config.source, type=config.type, uri=config.uri)
            for name, config in sources.items()
            if isinstance(config, ConfigBox)
        }

        return DataIngestionConfig(
            sources=sources,
            outdir=self.artifact_root
            // DATA_DIRECTORY_NAME
            // f"{INGESTED_DATA_FOLDER}_{DATA_DIRECTORY_NAME}",
        )

    @staticmethod
    def get_data_schema(name: str) -> DataSchema:
        file = os.path.join(SCHEMA_DIR, f"{name}.yaml")
        if os.path.exists(file):
            schema: dict = load_yaml(file, box=False)
            return DataSchema(
                name=name,
                schema=schema.columns,
                train=schema.train,
                test=schema.test,
                features=schema.features,
                target=schema.target,
            )
        else:
            e = f"Schema of dataset '{name}' not found"
            logger.error(e)
            raise Exception(e)

    @typechecked
    def get_data_validation_config(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationConfig:
        config: ConfigBox = self.config.data.validation
        input_dir = data_ingestion_artifact.path
        schemas: dict[str, DataSchema] = dict()
        for name in data_ingestion_artifact.names:
            if (input_dir / name).exists():
                schemas[name] = DataSchema(name=name, path=input_dir // name)
            else:
                e = DataNotFoundError(str(input_dir / name))
                logger.error(e)
                raise e

        if hasattr(config, "outdir"):
            target_dir = self.artifact_root // config.outdir
        else:
            target_dir = self.artifact_root // DATA_DIRECTORY_NAME

        if not input_dir.exists():
            e = DirectoryNotFoundError(input_dir, "Data hasn't ingested")
            logger.error(e)
            raise Exception(e)

        data_drift_config = None
        if config.get("data_drift", None):
            data_drift_config = DataDriftConfig(
                n_splits=config.data_drift.n_splits,
                n_iterations=config.data_drift.n_iterations,
                dimension=config.data_drift.dimension,
            )

        return DataValidationConfig(
            outdir=target_dir,
            report_name=config.get("report_name", REPORT_NAME),
            indir=input_dir,
            statistics=config.statistics,
            data_drift=data_drift_config,
            schemas=schemas,
        )

    @typechecked
    def get_data_transformation_config(
        self,
        data_validation_artifact: DataValidationArtifact,
        transform_config: ConfigBox,
    ) -> DataTransformationConfig:
        splitter = False
        if transform_config.get("splitter", None):
            try:
                splitter = DataSplitConfig(
                    type=transform_config.splitter.type,
                    n_splits=transform_config.splitter.n_splits,
                    labels=transform_config.splitter.get("labels", None),
                )
            except Exception as e:
                e = ConfigurationError(message=e)
                logger.error(e)
                raise e

        triplet_config = False
        if transform_config.get("triplet", None):
            try:
                triplet_config = TripletDataConfig(
                    anchor_col=transform_config.triplet.anchor_column,
                    sample_col=transform_config.triplet.sample_column,
                    n_negatives=transform_config.triplet.n_negatives,
                    n_samples=transform_config.triplet.n_samples,
                    reversed=transform_config.triplet.get("reversed", False),
                )
            except Exception as e:
                e = ConfigurationError(message=e)
                logger.error(e)
                raise e

        return DataTransformationConfig(
            outdir=self.artifact_root // DATA_DIRECTORY_NAME // TRANSFORM_DIR_NAME,
            indir=data_validation_artifact.valid_outdir,
            schemas=data_validation_artifact.schemas,
            splitter=splitter,
            triplet=triplet_config,
            pairwise=transform_config.get("pairwise", False),
            zero_shot=transform_config.get("zero-shot", False),
            augmentations=transform_config.get("augmentations", False),
            cache_intermediate=transform_config.get("cache-intermediate", False),
        )

    @typechecked
    def get_model_training_config(
        self, data_validation_artifact: DataValidationArtifact
    ) -> MultiModelTrainingConfig | Iterator[ModelTrainingConfig]:
        target_dir = self.artifact_root // MODELS_DIRECTORY_NAME
        model_configs = dict()
        for model_name, model_config in self.config.models.items():
            transformation = self.get_data_transformation_config(
                data_validation_artifact=data_validation_artifact,
                transform_config=model_config.transforms,
            )

            training_params = ModelTrainingConfig(
                name=model_name,
                outdir=target_dir // model_name,
                type=model_config.type,
                model=model_config.model,
                transformation=transformation,
                dataloader_pin_memory=model_config.get(
                    "dataloader-pin-memory",
                    self.config.get("dataloader-pin-memory", False),
                ),
                seed=model_config.get("seed", self.config.get("seed", 1234)),
                optimizer=model_config.get(
                    "optimizer", self.config.get("optimizer", "")
                ),
                max_grad_norm=model_config.get(
                    "max-grad-norm", self.config.get("max-grad-norm", "")
                ),
                n_epochs=model_config.get("n-epochs", self.config.get("n-epochs", 1)),
                learning_rate=model_config.get(
                    "learning-rate", self.config.get("learning-rate", None)
                ),
                gradient_accumulation_steps=model_config.get(
                    "gradient-accumulation-steps",
                    self.config.get("gradient-accumulation-steps", None),
                ),
                weight_decay=model_config.get(
                    "weight-decay", self.config.get("weight-decay", None)
                ),
                gradient_checkpointing=model_config.get(
                    "gradient-checkpointing",
                    self.config.get("gradient-checkpointing", False),
                ),
                train_file_path=transformation.outdir / "train.py",
                train_batch_size=model_config.get(
                    "train-batch-size", self.config.get("train-batch-size", None)
                ),
                valid_file_path=None,
                valid_batch_size=model_config.get(
                    "valid-batch-size", self.config.get("valid-batch-size", None)
                ),
                eval_strategy=model_config.get(
                    "eval-strategy", self.config.get("eval-strategy", "no")
                ),
                eval_steps=model_config.get(
                    "eval-steps", self.config.get("eval-steps", None)
                ),
                eval_epochs=model_config.get(
                    "eval-epochs", self.config.get("eval-epochs", None)
                ),
                logging_strategy=model_config.get(
                    "logging-strategy", self.config.get("logging-strategy", "no")
                ),
                logging_steps=model_config.get(
                    "logging-steps", self.config.get("logging-steps", None)
                ),
                logging_epochs=model_config.get(
                    "logging-epochs", self.config.get("logging-epochs", None)
                ),
                save_strategy=model_config.get(
                    "save-strategy", self.config.get("save-strategy", "no")
                ),
                save_steps=model_config.get(
                    "save-steps", self.config.get("save-steps", None)
                ),
                save_epochs=model_config.get(
                    "save-epochs", self.config.get("save-epochs", None)
                ),
                scheduler_type=model_config.get(
                    "scheduler-type", self.config.get("scheduler-type", "cosine")
                ),
                warmup_ratio=model_config.get(
                    "warmup-ratio", self.config.get("warmup-ratio", None)
                ),
                max_length=model_config.get(
                    "max-length", self.config.get("max-length", 640)
                ),
                padding=model_config.get(
                    "padding", self.config.get("padding", "longest")
                ),
                fold=model_config.get("fold", self.config.get("fold", -1)),
            )

            json_path = training_params.outdir / "training_params.json"
            save_json(training_params.model_dump(mode="json"), json_path)
            yield model_name, json_path
            model_configs[model_name] = json_path

        return MultiModelTrainingConfig(outdir=target_dir, models=model_configs)

    @typechecked
    def get_model_inference_config(
        self, model_training_artifact: MultiModelTrainingArtifact
    ) -> MultiModelInferenceConfig:
        target_dir = self.artifact_root // "inference"
        model_configs = dict()
        for model_name, model_config in self.config.models.items():
            inference_params = ModelInferenceConfig(
                name=model_name,
                outdir=target_dir // model_name,
                type=model_config.type,
                model_path=model_config.model,
                batch_size=model_config.get("inference-batch-size")
                or self.config.get("inference-batch-size")
                or 8,
                train_path=model_training_artifact.models[model_name].model_path,
                max_length=model_config.get("max-lenght")
                or self.config.get("max-length")
                or 256,
                tta=False,
                ensmeble_weight=1,
            )
            json_path = inference_params.outdir / "inference_params.json"
            save_json(inference_params.model_dump(mode="json"), json_path)
            model_configs[model_name] = json_path

        return MultiModelInferenceConfig(outdir=target_dir, models=model_configs)
