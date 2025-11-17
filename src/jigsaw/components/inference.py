from ..core import FilePath, ModelInferenceConfig, ModelInferenceArtifact
from ..utils.common import load_json, save_csv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import torch


class ModelInferenceComponent:
    def __init__(self, config: FilePath | ModelInferenceConfig):
        if isinstance(config, FilePath):
            config = ModelInferenceConfig(**load_json(config))

        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.config.train_path), device_map="auto"
        )
        self.device = self.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.config.train_path))

    def __call__(self, data: pd.DataFrame) -> ModelInferenceArtifact:
        prompts = (data["rule"] + "[SEP]" + data["body"]).tolist()

        outputs = np.zeros(len(prompts), dtype=np.float64)
        for idx in range(0, len(prompts), self.config.batch_size):
            inputs = self.tokenizer(
                prompts[idx : idx + self.config.batch_size],
                padding="longest",
                truncation=True,
                max_length=self.config.max_length,
            )
            inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
            outputs[idx : idx + self.config.batch_size] = (
                softmax(self.model(**inputs).logits, dim=-1)
                .cpu()
                .detach()
                .numpy()[:, 1]
            )

        data["prediction"] = outputs
        save_csv(data, self.config.outdir / "prediction.csv")
        return ModelInferenceArtifact(
            name=self.config.name,
            prediction_file_path=self.config.outdir / "prediction.csv",
        )
