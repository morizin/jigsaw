import os
from typing import TypeVar
from src.jigsaw.utils.common import load_yaml 
from src.jigsaw.constants import *
from pathlib import Path
from src.jigsaw.entity.config_entity import (DataIngestionConfig, DataSource)
from src.jigsaw import logger
from src.jigsaw.entity.common import (Directory)
from box import ConfigBox

FilePath = TypeVar("FilePath", str, Path)

class ConfigurationManager:
    def __init__(self, config_path : FilePath = CONFIG_FILE_PATH,
                 params_path : FilePath = PARAMS_FILE_PATH,
                 schema_path : FilePath = SCHEMA_FILE_PATH):
        self.config = load_yaml(config_path)
        self.params = load_yaml(params_path)
        self.schema = load_yaml(schema_path)
        self.artifact_root = Directory(path = self.config.artifact_root)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion = self.config.data_ingestion

        sources = []
        names = []
        for (name, config) in self.config.data_ingestion.items():
            if isinstance(config, ConfigBox):
                sources.append(
                        DataSource(
                            source = config.source,
                            name = config.name,
                            type = config.type
                        )
                )
                names.append(name)


        return DataIngestionConfig(
                sources = sources,
                names = names,  
                outdir = Directory(path = self.artifact_root.path/data_ingestion.outdir) 
        )
