from src.jigsaw.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.jigsaw.pipelines.data_validation_pipeline import DataValidationPipeline
from src.jigsaw.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.jigsaw.pipelines.model_training_pipeline import ModelTrainingPipeline
from src.jigsaw import logger

try:
    logger.info("Kicking off Data Ingesion Pipeline")
    DataIngestionPipeline().kickoff()
    logger.info("Data Ingesion Pipeline Completed")
except Exception as e:
    logger.error(f"Error during data ingestion {e}")

try:
    logger.info("Kicking off Data Validation Pipeline")
    DataValidationPipeline().kickoff()
    logger.info("Data Validation Pipeline Completed")
except Exception as e:
    logger.error(f"Error during data validation {e}")
    raise e

try:
    logger.info("Kicking off Data Transformation Pipeline")
    DataTransformationPipeline().kickoff()
    logger.info("Data Transformation Pipeline Completed")
except Exception as e:
    logger.error(f"Error during data transformation {e}")
    raise e

try:
    logger.info("Kicking off Model Training Pipeline")
    ModelTrainingPipeline().kickoff()
    logger.info("Model Training Pipeline Completed")
except Exception as e:
    logger.error(f"Error during model training pipeline {e}")
    raise e

