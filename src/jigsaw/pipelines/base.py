from src.jigsaw.config import ConfigurationManager
from ..core import (
    DataIngestionConfig,
    DataValidationConfig,
    DataIngestionArtifact,
    DataValidationArtifact,
    ModelTrainingArtifact,
    MultiModelTrainingArtifact,
)

from typeguard import typechecked
from .. import logger

from ..components.data import DataIngestionComponent, DataValidationComponent
from ..components.train import ModelTrainingComponent


class BasePipeline:
    def __init__(self):
        try:
            logger.info("Configuring....")
            self.config: ConfigurationManager = ConfigurationManager()
        except Exception as e:
            logger.error(f"Error Configuring... {e}")

    @typechecked
    def do_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Data Ingestion ....")
            data_ingestion_config: DataIngestionConfig = (
                self.config.get_data_ingestion_config()
            )

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionComponent(
                data_ingestion_config
            )()

            logger.info("Data Ingesion Completed")
            return data_ingestion_artifact

        except Exception as e:
            logger.error(f"Error during data ingestion {e}")
            raise e

    @typechecked
    def do_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logger.info("Data Validation ....")
            data_validation_config: DataValidationConfig = (
                self.config.get_data_validation_config(data_ingestion_artifact)
            )

            data_validation_artifact: DataValidationArtifact = DataValidationComponent(
                data_validation_config
            )()

            logger.info("Data Validation Completed")
            return data_validation_artifact

        except Exception as e:
            logger.error(f"Error during data validation {e}")
            raise e

    def do_model_training(self, data_validation_artifact: DataValidationArtifact):
        logger.info("Kicking off Training Pipeline")
        try:
            logger.info("Model Training ...")

            model_training_configs = iter(
                self.config.get_model_training_config(data_validation_artifact)
            )
            model_training_artifacts = dict()

            try:
                while True:
                    model_name, model_training_config = next(model_training_configs)
                    logger.info(f"Model Training : {model_name}")
                    model_training_artifacts[model_name] = ModelTrainingComponent(
                        model_training_config,
                    )()
            except StopIteration as e:
                model_training_configs = e.value

            logger.info("Model Training Completed")
            logger.info("Training Pipeline Completed")
            return MultiModelTrainingArtifact(
                outdir=model_training_configs.outdir, models=model_training_artifacts
            )

        except Exception as e:
            logger.error(f"Error during model training {e}")
            raise e

    def kickoff(
        self,
    ):
        logger.info("Kicking off Base Pipeline")
        data_ingestion_artifact: DataIngestionArtifact = self.do_data_ingestion()
        data_validation_artifact: DataValidationArtifact = self.do_data_validation(
            data_ingestion_artifact
        )

        model_training_artifact: ModelTrainingArtifact = self.do_model_training(
            data_validation_artifact
        )
        logger.info("Base Pipeline Completed")
        print(model_training_artifact)
        return model_training_artifact
