# from jigsaw.config import ConfigurationManager
from ..core import (
    DataTransformationConfig,
    DataTransformationArtifact,
    DataValidationArtifact,
    MultiModelTrainingConfig,
    MultiModelTrainingArtifact,
)

from jigsaw import logger
from ..components.data import DataTransformationComponent
from ..components.train import ModelTrainingComponent
from typeguard import typechecked


class TrainingPipeline:
    def __init__(
        self,
        config,
    ):
        self.config = config

    @typechecked
    def do_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            logger.info("Data Transformation ...")
            data_transformation_config: DataTransformationConfig = (
                self.config.get_data_transformation_config(data_validation_artifact)
            )

            data_transformation_artifact = DataTransformationComponent(
                data_transformation_config
            )()
            logger.info("Data Transformation Completed")
            return data_transformation_artifact

        except Exception as e:
            logger.error(f"Error during data transformation {e}")
            raise e

    @typechecked
    def do_model_training(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> MultiModelTrainingArtifact:
        try:
            logger.info("Model Training ...")
            model_training_configs: MultiModelTrainingConfig = (
                self.config.get_model_training_config(data_transformation_artifact)
            )

            model_training_artifacts = dict()
            for (
                model_name,
                model_training_config,
            ) in model_training_configs.models.items():
                model_training_artifacts[model_name] = ModelTrainingComponent(
                    model_training_config,
                )()
            logger.info("Model Training Completed")

            return MultiModelTrainingArtifact(
                outdir=model_training_configs.outdir, models=model_training_artifacts
            )

        except Exception as e:
            logger.error(f"Error during model training {e}")
            raise e

    def kickoff(
        self,
        data_validation_artifact: DataValidationArtifact,
    ) -> MultiModelTrainingArtifact:
        logger.info("Kicking off Training Pipeline")
        data_transformation_artifact: DataTransformationArtifact = (
            self.do_data_transformation(data_validation_artifact)
        )
        model_training_artifact: MultiModelTrainingArtifact = self.do_model_training(
            data_transformation_artifact
        )
        logger.info("Training Pipeline Completed")
        return model_training_artifact
