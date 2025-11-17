from ..core import (
    MultiModelInferenceConfig,
    ModelInferenceArtifact,
    MultiModelInferenceArtifact,
)
from .. import logger
import pandas as pd
from ..components.inference import ModelInferenceComponent


class PredictorPipeline:
    def __init__(self, config, model_training_artifact):
        self.config = config

        logger.info("Model Inferencing ...")
        self.config: MultiModelInferenceConfig = self.config.get_model_inference_config(
            model_training_artifact
        )

        self.model_inference_component = dict()
        for model_name, model_inference_config in self.config.models.items():
            self.model_inference_component[model_name] = ModelInferenceComponent(
                model_inference_config,
            )

    def __call__(self, data: pd.DataFrame) -> MultiModelInferenceArtifact:
        model_inference_artifacts = dict()
        try:
            for model_name, predict_component in self.model_inference_component.items():
                model_inference_artifact: ModelInferenceArtifact = predict_component(
                    data.copy()
                )
                model_inference_artifacts[model_name] = model_inference_artifact

            logger.info("Model Inferencing Completed")
            return MultiModelInferenceArtifact(
                outdir=self.config.outdir, models=model_inference_artifacts
            )

        except Exception as e:
            logger.error(f"Error during model training {e}")
            raise e
