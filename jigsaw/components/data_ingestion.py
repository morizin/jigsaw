from jigsaw.utils.common import load_yaml
from jigsaw.config.config import DataIngestionConfig
from jigsaw.types import Directory

class DataIngestionComponent:
     def __init__ (self, config_path):
         self.config_path = config_path
         self.config = load_yaml(self.config_path)
         assert hasattr(self.config, "data-ingestion"), "Data Ingestion isn't Configured"
         self.config = self.config.data_ingestion
         self.dataset = self.load_config("raw")

     def load_config(self, dataset_name : str):
         assert hasattr(self.config, dataset_name), f"Dataset Named \"{dataset_name}\" doesn't exist"
         dataset = getattr(self.config, dataset_name)
         return DataIngestionConfig(
            source = dataset.source,
            type = dataset.type,
            name = dataset.name,
            outdir = Directory(path = dataset.outdir)
        )

     def download_file(self):
         pass
