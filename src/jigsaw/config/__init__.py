import os
from ..utils.common import seed_everything, load_yaml, load_json, get_hw_details
from ..constants import (
    CONFIG_FILE_PATH,
    DATA_DIRECTORY_NAME,
    MODELS_DIRECTORY_NAME,
    SCHEMA_DIR,
)
from ..constants.data import INGESTED_DATA_FOLDER, REPORT_NAME
from ..errors import DataNotFoundError, DirectoryNotFoundError

from ..core import (
    DataIngestionConfig,
    DataSource,
    DataValidationConfig,
    DataSchema,
    TripletDataConfig,
    DataTransformationConfig,
    DataSplitConfig,
    EngineConfig,
    TokenizerConfig,
    ModelTrainingConfig,
    FilePath,
    Directory,
    DataIngestionArtifact,
)
from .. import logger, PROJECT_NAME, timestamp
from datetime import datetime
from box import ConfigBox
from typeguard import typechecked


class ConfigurationManager:
    @typechecked
    def __init__(
        self,
        config_path: FilePath = CONFIG_FILE_PATH,
    ):
        self.config: ConfigBox = load_yaml(config_path)
        seed_everything(self.config.seed)

        n_procs, n_threads = get_hw_details()
        self.config.n_procs = n_procs
        self.config.n_threads = n_threads

        logger.info(f"Running on {n_procs} CPU(s) with {n_threads} Thread(s) each")

        self.artifact_root: Directory = (
            Directory(path=self.config.artifact_root)  # // timestamp
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        sources = self.config.data.sources

        sources: dict[str, DataSource] = {
            name: DataSource(source=config.source, type=config.type, name=config.name)
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

        return DataValidationConfig(
            outdir=target_dir,
            report_name=config.get("report_name", REPORT_NAME),
            indir=input_dir,
            statistics=config.statistics,
            data_drift=config.data_drift,
            schemas=schemas,
        )

    @typechecked
    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transform = self.config.data_transformation
        if hasattr(data_transform, "splitter") and data_transform.splitter:
            try:
                split_config = self.params[data_transform.splitter]
                splitter = DataSplitConfig(
                    type=split_config.type,
                    nsplits=split_config.nsplits,
                    random_state=split_config.random_state,
                )

                if hasattr(split_config, "labels"):
                    splitter.labels = split_config.labels
            except KeyError:
                logger.error(f"Splitter '{data_transform.splitter} doesn't exist")
                splitter = None
            except Exception as e:
                raise e

        else:
            splitter = None

        if hasattr(data_transform, "triplet") and data_transform.triplet:
            triplet_config = TripletDataConfig(
                ntriplets=data_transform.triplet.ntriplets,
                nsamples=data_transform.triplet.nsamples,
                random_state=self.params.SEED,
            )
        else:
            triplet_config = None

        status_file = load_json(
            self.artifact_root.path
            / os.path.join(self.config.data_validation.outdir, "status.json")
        )

        features = dict()
        targets = dict()
        names = []
        for name, schema in self.schema.items():
            names.append(name)
            if name in status_file:
                for value in status_file[name].values():
                    if value["data_redundancy"]:
                        try:
                            features[name] = schema.features
                            targets[name] = schema.target
                        except Exception as e:
                            logger.error(e)
                            features[name] = None
                            targets[name] = None

        if not hasattr(data_transform, "urlparse"):
            data_transform.urlparse = False

        if not hasattr(data_transform, "wash"):
            data_transform.wash = False

        if not hasattr(data_transform, "zero"):
            data_transform.zero = False

        if not hasattr(data_transform, "pairwise"):
            data_transform.pairwise = False

        final_dir = ""
        if len(features):
            final_dir = "cleaned_" + final_dir

        if data_transform.urlparse:
            final_dir = "parse_" + final_dir

        if data_transform.wash:
            final_dir = "washed_" + final_dir

        if data_transform.zero:
            final_dir = "zero_" + final_dir

        if triplet_config:
            final_dir = "triplet_" + final_dir

        if data_transform.pairwise:
            final_dir = "pairwise_" + final_dir

        if splitter:
            final_dir = "folded_" + final_dir

        self.final_dir = final_dir
        self.names = names

        return DataTransformationConfig(
            outdir=Directory(path=self.artifact_root.path / data_transform.outdir),
            indir=Directory(
                path=self.artifact_root.path / self.config.data_ingestion.outdir
            ),
            datasets=names,
            splitter=splitter,
            features=features,
            targets=targets,
            urlparse=data_transform.urlparse,
            wash=data_transform.wash,
            triplet=triplet_config,
            zero=data_transform.zero,
            pairwise=data_transform.pairwise,
            final_dir=final_dir,
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
