from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from transformers import AutoTokenizer
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
    DebertaV2TokenizerFast,
)
from pandas.api.types import is_string_dtype
from src.jigsaw.utils.common import read_csv
from src.jigsaw.entity.config_entity import ClassifierConfig
from typing import Dict
from typeguard import typechecked
import torch


class ClassifierDataset(Dataset):
    @typechecked
    def __init__(
        self,
        config: ClassifierConfig,
        data: DataFrame | str,
        tokenizer: DebertaV2Tokenizer | DebertaV2TokenizerFast | str,
    ):
        if isinstance(data, str):
            data = read_csv(data)

        if isinstance(data, DataFrame):
            self.data = data
        else:
            error = f"'data' can be either str or pd.DataFrame. 'data' has type {type(data).__name__}"
            logger.error(error)
            raise Exception(error)

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            
        if isinstance(tokenizer, (DebertaV2Tokenizer, DebertaV2TokenizerFast)):
            self.tokenizer = tokenizer
        else:
            error = f"'tokenizer' can be either str, DebertaV2Tokenizer, DebertaV2TokenizerFast. 'tokenizer' has type {type(tokenizer).__name__}"
            logger.error(error)
            raise Exception(error)

        for (col, dtype) in data.dtypes.items():
            if is_string_dtype(dtype):
                data[col] = data[col] + data[col].apply(url_to_semantics)

        self.completion = data.apply(build_prompt, axis=1).to_list()
        
        self.encoding = self.tokenizer(
            self.completion, truncation=config.truncation, padding = config.padding, max_length=config.max_length
        )

        if isinstance(config.labels, str):
            config.labels = [config.labels]

        if any([col in data.columns for col in config.labels]):
            self.labels = data[config.labels].to_numpy()
        else:
            self.labels = None

    def __len__(self,) -> int:
        assert len(self.encoding['input_ids']) == len(self.labels), f"Input and Output length mismatch {len(self.encoding)} != {len(self.labels)}"
        return len(self.encoding['input_ids'])
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]: 
        items = {key : torch.tensor(value[idx]) for (key, value) in self.encoding.items()}
        if self.labels is not None:
            items['label'] = torch.tensor(self.labels[idx].flatten())
        return items
