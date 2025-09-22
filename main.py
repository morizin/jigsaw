from src.jigsaw.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.jigsaw.pipelines.data_validation_pipeline import DataValidationPipeline
from src.jigsaw import logger

try:
    logger.info("Kicking off DataIngesionPipeline")
    DataIngestionPipeline().kickoff()
    logger.info("DataIngesionPipeline Completed")
except Exception as e:
    logger.error(f"Error during data ingestion {e}")

try:
    logger.info("Kicking off DataValidationPipeline")
    DataValidationPipeline().kickoff()
    logger.info("DataValidationPipeline Completed")
except Exception as e:
    logger.error(f"Error during data validation {e}")
