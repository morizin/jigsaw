from pathlib import Path
from ...utils.common import load_yaml
from ...entity.config_entity import DataIngestionConfig
from ...entity.common import Directory
import kagglehub
import subprocess
import os
from ... import logger

class DataIngestionComponent:
    def __init__ (self, config: DataIngestionConfig):
        self.config = config
        print("==============================")
        print("|      Available Dataset     |")
        print("==============================")
        for i in self.config.names:
            print(f"\t{i}")
        print("==============================")

    def download_all(self):
        for datasource, name in zip(self.config.sources, self.config.names):
            if datasource.source.lower().strip() == 'kaggle':
                if datasource.type.lower().strip() == 'competition':
                    path = kagglehub.competition_download(datasource.name,) 
                    target_path = Directory(path = self.config.outdir.path / name)
                    logger.info(f"Downloading {datasource.name} competition dataset to {target_path.path.as_posix()}")
                    subprocess.call(f'mv {path.rstrip("/") + "/*"} {target_path.path.as_posix().rstrip("/") + "/"}', shell = True)
                    subprocess.call(f'rm -rf {path}'.split())

