from src.jigsaw.config import ConfigurationManager
from ..core import (
    FilePath,
    DataValidationArtifact,
    ModelTrainingArtifact,
    MultiModelTrainingArtifact,
)
from .data import DataPipeline
from .train import TrainingPipeline
from .inference import PredictorPipeline
from typeguard import typechecked
from .. import logger


class BasePipeline:
    def __init__(self):
        try:
            logger.info("Configuring....")
            self.config: ConfigurationManager = ConfigurationManager("config.yaml")
        except Exception as e:
            logger.error(f"Error Configuring... {e}")

    @typechecked
    def do_data_pipeline(
        self,
    ) -> DataValidationArtifact:
        pipeline = DataPipeline(self.config)
        return pipeline.kickoff()

    @typechecked
    def do_training_pipeline(
        self, data_config: DataValidationArtifact
    ) -> MultiModelTrainingArtifact:
        pipeline = TrainingPipeline(self.config)
        return pipeline.kickoff(data_config)

    @typechecked
    def do_prediction_pipeline(
        self,
        model_training_artifact: MultiModelTrainingArtifact,
    ):
        predictor = PredictorPipeline(self.config, model_training_artifact)
        return predictor()

    def kickoff(
        self,
    ):
        logger.info("Kicking off Base Pipeline")
        data_validation_artifact: DataValidationArtifact = self.do_data_pipeline()
        model_training_artifact: ModelTrainingArtifact = self.do_training_pipeline(
            data_validation_artifact
        )
        logger.info("Base Pipeline Completed")
        return model_training_artifact
