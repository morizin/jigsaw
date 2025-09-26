from src.jigsaw.config.config import ConfigurationManager
from src.jigsaw.components.data.transformation import DataTransformationComponent

class DataTransformationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_data_transformation_config()
        self.comp = DataTransformationComponent(self.config)

    def kickoff(self):
        self.comp()
