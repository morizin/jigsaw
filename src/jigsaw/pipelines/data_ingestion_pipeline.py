from src.jigsaw.components.data.ingestion import DataIngestionComponent
from src.jigsaw.config.config import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self, ):
        self.config = ConfigurationManager().get_data_ingestion_config()
    
    def kickoff(self):
        DataIngestionComponent(self.config).download_all()

