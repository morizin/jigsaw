import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from ...entity.config_entity import ModelTrainingConfig 

def get_deberta_model(config : ModelTrainingConfig) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(config.engine.model_name,
                                                              trust_remote_code = True,
                                                              num_labels = 2)
