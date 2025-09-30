import os
from typing import TypeVar
from ..utils.common import load_yaml, load_json
from ..constants import *
from pathlib import Path
from ..entity.config_entity import (
    DataIngestionConfig,
    DataSource,
    DataValidationConfig,
    DataSchema,
    TripletDataConfig,
    DataTransformationConfig,
    DataSplitParams,
    EngineParams,
    TokenizerParams,
    ModelTrainingConfig,
)
from .. import logger
from ..entity.common import FilePath, Directory
from box import ConfigBox
from typeguard import typechecked

class ConfigurationManager:

    @typechecked
    def __init__(
        self,
        config_path: FilePath = CONFIG_FILE_PATH,
        params_path: FilePath = PARAMS_FILE_PATH,
        schema_path: FilePath = SCHEMA_FILE_PATH,
    ):
        self.config = load_yaml(config_path)
        self.params = load_yaml(params_path)
        self.schema = load_yaml(schema_path)
        self.artifact_root = Directory(path=self.config.artifact_root)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion = self.config.data_ingestion

        sources = []
        names = []
        for name, config in self.config.data_ingestion.items():
            if isinstance(config, ConfigBox):
                sources.append(
                    DataSource(source=config.source, name=config.name, type=config.type)
                )
                names.append(name)

        return DataIngestionConfig(
            sources=sources,
            names=names,
            outdir=Directory(path=self.artifact_root.path / data_ingestion.outdir),
        )

    @typechecked
    def get_data_schema(self, name: str) -> DataSchema:
        if hasattr(self.schema, name):
            schema = self.schema[name]
            return DataSchema(
                name=name,
                schema=schema.columns.to_dict(),
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
    def get_data_validation_config(self):
        config = self.config.data_validation
        schemas = []
        self.names = self.config.data_ingestion
        for name in self.names:
            if name == "outdir":
                continue
            if name in os.listdir(
                self.artifact_root.path / self.config.data_ingestion.outdir
            ):
                schemas.append(self.get_data_schema(name))
            else:
                logger.error(f'Data "{name}" doesn\'t exists)')

        target_dir = Directory(path=self.artifact_root.path / config.outdir)
        input_dir = self.artifact_root.path / self.config.data_ingestion.outdir
        if not input_dir.exists():
            logger.error("Data Validation : Data hasn't ingested")
            raise Exception("Data Validation : Data hasn't ingested")

        return DataValidationConfig(
            outdir=target_dir,
            indir=Directory(path=input_dir),
            statistics=config.statistics,
            schemas=schemas,
        )

    @typechecked
    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transform = self.config.data_transformation
        if hasattr(data_transform, "splitter") and data_transform.splitter:
            try:
                split_config = self.params[data_transform.splitter]
                splitter = DataSplitParams(
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
                        except:
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

        engine_params = EngineParams(
            model_name=engine_config.model_name,
            nepochs=engine_config.nepochs,
            learning_rate=engine_config.learning_rate,
            train_batch_size = engine_config.train_batch_size,
            valid_batch_size = engine_config.get("valid_batch_size", None) ,
            gradient_accumulation_steps=engine_config.get(
                "gradient_accumulation_steps", 1
            ),
            weight_decay=engine_config.get("weight_decay", None),
            warmup_ratio=engine_config.get("warmup_ratio", None),
            tokenizer=TokenizerParams(
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
            outdir=Directory(path=config.outdir),
            indir = Directory(path = config.indir),
            fold=config.get("fold", -1),
            engine=engine_params,
            schemas=schemas,
        )
