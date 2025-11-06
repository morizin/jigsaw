from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from transformers import AutoTokenizer
from typing import Dict
from typeguard import typechecked
import torch
from .prompts import zero_shot_chat_prompt
from ... import logger
from ...utils.common import load_csv
from ...schema.config_entity import ModelTrainingConfig


class ClassifierDataset(Dataset):
    @typechecked
    def __init__(
        self,
        config: ModelTrainingConfig,
        data: DataFrame | str,
    ):
        self.config = config
        if isinstance(data, str):
            data = load_csv(data)

        tokenizer_config = config.engine.tokenizer

        if isinstance(data, DataFrame):
            self.data = data
        else:
            error = f"'data' can be either str or pd.DataFrame. 'data' has type {type(data).__name__}"
            logger.error(error)
            raise Exception(error)

        self.tokenizer = AutoTokenizer.from_pretrained(config.engine.model_name)

        self.completion = data.apply(
            zero_shot_chat_prompt, axis=1, args=(self.tokenizer,)
        ).to_list()

        self.encoding = self.tokenizer(
            self.completion,
            truncation=tokenizer_config.truncation,
            padding=tokenizer_config.padding,
            max_length=tokenizer_config.max_length,
        )

        target = config.schemas[0].target
        if isinstance(target, str):
            target = [target]

        if any([col in data.columns for col in target]):
            self.labels = data[target].to_numpy()
        else:
            self.labels = None

    def __len__(
        self,
    ) -> int:
        assert len(self.encoding["input_ids"]) == len(self.labels), (
            f"Input and Output length mismatch {len(self.encoding['input_ids'])} != {len(self.labels)}"
        )
        return len(self.encoding["input_ids"])

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        items = {
            key: torch.tensor(value[idx]) for (key, value) in self.encoding.items()
        }
        if self.labels is not None:
            items["labels"] = torch.tensor(self.labels[idx, 0])
        return items


class LLMDataset:
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        row = self.data.iloc[idx]

        inputs = {
            "input_ids": row["input_ids"],
        }
        return inputs
