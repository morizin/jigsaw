# from jigsaw.config import ConfigurationManager
from ..core import (
    DataIngestionConfig,
    DataValidationConfig,
    DataIngestionArtifact,
    DataValidationArtifact,
)

from jigsaw import logger
from ..components.data import DataIngestionComponent, DataValidationComponent


class DataPipeline:
    def __init__(self, config):
        self.config = config

    # @typechecked
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

    # @typechecked
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

    def kickoff(
        self,
    ):
        logger.info("Kicking off Data Pipeline")
        data_ingestion_artifact: DataIngestionArtifact = self.do_data_ingestion()
        data_validation_artifact: DataValidationArtifact = self.do_data_validation(
            data_ingestion_artifact
        )
        logger.info("Data Pipeline Completed")
        return data_validation_artifact
