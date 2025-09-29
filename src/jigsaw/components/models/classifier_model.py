import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from ...entity.config_entity import ModelConfig

class ClassifierModel(nn.Module):
    
    def __init__(self, config: ModelConfig):
        self.backbone = AutoModelForSequenceClassification.from_pretrained(config.model_name, trust_remote_code = True)
        self.backbone.classifier = nn.Linear(self.config.classifier.in_features, 1)
        del self.backbone.dropout

    def forward(self, inputs):
        return self.backbone(**inputs)

if __name__ == "__main__":
