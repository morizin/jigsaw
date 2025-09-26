import os
from typing import TypeVar
from src.jigsaw.utils.common import load_yaml, load_json
from src.jigsaw.constants import *
from pathlib import Path
from src.jigsaw.entity.config_entity import (
    DataIngestionConfig,
    DataSource,
    DataValidationConfig,
    DataSchema,
    DataTransformationConfig,
    DataSplitParams,
)
from src.jigsaw import logger
from src.jigsaw.entity.common import Directory
from box import ConfigBox

FilePath = TypeVar("FilePath", str, Path)


class ConfigurationManager:
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

    def get_data_validation_config(self):
        config = self.config.data_validation
        schemas = []

        for name, schema in self.schema.items():
            if name in os.listdir(
                self.artifact_root.path / self.config.data_ingestion.outdir
            ):
                schemas.append(
                    DataSchema(
                        name=name,
                        schema=schema.columns.to_dict(),
                        train=schema.train,
                        test=schema.test,
                        features=schema.features,
                        target=schema.target,
                    )
                )
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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        split_config = self.params.splitter
        data_transform = self.config.data_transformation
        splitter = DataSplitParams(
            type=split_config.type,
            nsplits=split_config.nsplits,
            random_state=split_config.random_state,
        )
        if hasattr(split_config, 'labels'):
            splitter.labels = split_config.labels

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
                            

        return DataTransformationConfig(
            outdir=Directory(path=self.artifact_root.path / data_transform.outdir),
            indir=Directory(
                path=self.artifact_root.path / self.config.data_ingestion.outdir
            ),
            datasets=names,
            splitter=splitter,
            features=features,
            targets = targets,
            wash=data_transform.wash if hasattr(data_transform, "wash") else False,
            triplet=data_transform.triplet
            if hasattr(data_transform, "triplet")
            else False,
            zero=data_transform.zero if hasattr(data_transform, "zero") else False,
            pairwise=data_transform.pairwise
            if hasattr(data_transform, "pairwise")
            else False,
        )
