from src.jigsaw.config import ConfigurationManager
from ..core import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from ..components.data import (
    DataIngestionComponent,
    DataValidationComponent,
    DataTransformationComponent,
)
from typeguard import typechecked
from .. import logger


class BasePipelines:
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

    def kickoff(
        self,
    ):
        logger.info("Kicking off Base Pipeline")
        data_ingestion_artifact: DataIngestionArtifact = self.do_data_ingestion()
        data_validation_artifact: DataValidationArtifact = self.do_data_validation(
            data_ingestion_artifact
        )
        data_transformation_artifact = self.do_data_transformation(
            data_validation_artifact
        )
        print(data_transformation_artifact)
        logger.info("Base Pipeline Completed")

        #   try:
        #       logger.info("Kicking off Data Validation Pipeline")
        #       DataValidationPipeline().kickoff()
        #       logger.info("Data Validation Pipeline Completed")
        #   except Exception as e:
        #       logger.error(f"Error during data validation {e}")
        #       raise e

        #   try:
        #       logger.info("Kicking off Data Transformation Pipeline")
        #       DataTransformationPipeline().kickoff()
        #       logger.info("Data Transformation Pipeline Completed")
        #   except Exception as e:
        #       logger.error(f"Error during data transformation {e}")
        #       raise e

        #   try:
        #       logger.info("Kicking off Model Training Pipeline")
        #       ModelTrainingPipeline().kickoff()
        #       logger.info("Model Training Pipeline Completed")
        #   except Exception as e:
        #       logger.error(f"Error during model training pipeline {e}")
        #       raise e
