from src.jigsaw.config.config import ConfigurationManager
from src.jigsaw.components.data_validation import DataValidationComponent


class DataValidationPipeline:
    def __init__(
        self,
    ):
        self.config = ConfigurationManager().get_data_validation_config()
        self.comp = DataValidationComponent(self.config)

    def kickoff(
        self,
    ):
        self.comp.validate_all()
        if self.config.statistics:
            self.comp.get_statistics()
