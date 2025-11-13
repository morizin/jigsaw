import os
from ..utils.common import seed_everything, load_yaml, load_json, get_hw_details
from ..constants import (
    CONFIG_FILE_PATH,
    DATA_DIRECTORY_NAME,
    MODELS_DIRECTORY_NAME,
    SCHEMA_DIR,
)
from ..constants.data import INGESTED_DATA_FOLDER, REPORT_NAME, TRANSFORM_DIR_NAME
from ..errors import DataNotFoundError, DirectoryNotFoundError, ConfigurationError
from ..components.data.augmentation import Augmentor

from ..core import (
    DataIngestionConfig,
    DataSource,
    DataValidationConfig,
    DataSchema,
    DataDriftConfig,
    TripletDataConfig,
    DataTransformationConfig,
    DataSplitConfig,
    EngineConfig,
    TokenizerConfig,
    ModelTrainingConfig,
    FilePath,
    Directory,
    DataIngestionArtifact,
    DataValidationArtifact,
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
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationConfig:
        data_transform = self.config.data
        splitter = False
        if data_transform.get("splitter", None):
            try:
                splitter = DataSplitConfig(
                    type=data_transform.splitter.type,
                    n_splits=data_transform.splitter.n_splits,
                    labels=data_transform.splitter.get("labels", None),
                )
            except Exception as e:
                e = ConfigurationError(message=e)
                logger.error(e)
                raise e

        triplet_config = False
        if data_transform.get("triplet", None):
            try:
                triplet_config = TripletDataConfig(
                    anchor_col=data_transform.triplet.anchor_column,
                    sample_col=data_transform.triplet.sample_column,
                    n_negatives=data_transform.triplet.n_negatives,
                    n_samples=data_transform.triplet.n_samples,
                    reversed=data_transform.triplet.get("reversed", False),
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
            pairwise=data_transform.get("pairwise", False),
            zero_shot=data_transform.get("zero-shot", False),
            cache_intermediate=data_transform.get("cache-intermediate", False),
        )

    @typechecked
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        _ = self.get_data_transformation_config()
        try:
            engine_config = self.params[config.engine]
        except Exception as e:
            logger.error(f"TrainingArguments '{config.engine}' not found: {e}")
            raise e

        engine_params = EngineConfig(
            model_name=engine_config.model_name,
            nepochs=engine_config.nepochs,
            learning_rate=engine_config.learning_rate,
            train_batch_size=engine_config.train_batch_size,
            valid_batch_size=engine_config.get("valid_batch_size", None),
            gradient_accumulation_steps=engine_config.get(
                "gradient-accumulation-steps", 1
            ),
            weight_decay=engine_config.get("weight-decay", None),
            warmup_ratio=engine_config.get("warmup-ratio", None),
            tokenizer=TokenizerConfig(
                max_length=engine_config.tokenizer.max_length,
                truncation=engine_config.tokenizer.truncation,
                padding=engine_config.tokenizer.padding,
            ),
        )

        schemas = []
        for name in self.names:
            if f"{self.final_dir}{name}" in os.listdir(
                self.artifact_root.path / config.indir
            ):
                schema = self.get_data_schema(name)
                schema.name = f"{self.final_dir}{name}"
                if config.few_shot:
                    schema.features = list(schema.schema)
                del schema.schema
                schemas.append(schema)
            else:
                logger.error(f'Data "{name}" doesn\'t exists)')

        return ModelTrainingConfig(
            outdir=Directory(path=self.artifact_root.path / config.outdir),
            indir=Directory(path=self.artifact_root.path / config.indir),
            fold=config.get("fold", -1),
            engine=engine_params,
            schemas=schemas,
        )
