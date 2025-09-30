from ..config.config import ConfigurationManager
from ..components.engines.classifier_engine import ClassifierEngine

class ModelTrainingPipeline:
    def __init__(self, ) :
        self.config = ConfigurationManager().get_model_training_config()
        self.comp = ClassifierEngine(self.config)

    def kickoff(self):
        self.comp()
