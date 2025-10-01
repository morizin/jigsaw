import os
from typeguard import typechecked
import torch.nn as nn
from ...entity.config_entity import ModelTrainingConfig
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
import pandas as pd
from ... import logger
from pandas.core.frame import DataFrame
from ...utils.common import load_csv
from ..models.classifier_model import get_deberta_model
from ..dataset.classfier_dataset import ClassifierDataset

class ClassifierEngine:
    @typechecked
    def __init__(self, config : ModelTrainingConfig):
        train_data, valid_data = self.get_train_test_split(config)
        self.dataset = ClassifierDataset(config, train_data) 
    
        self.model = get_deberta_model(config)
        
        engine_config = config.engine
        self.training_args = TrainingArguments(
                output_dir = config.outdir.path,
                per_device_train_batch_size = engine_config.train_batch_size,
                gradient_accumulation_steps = engine_config.gradient_accumulation_steps, 
                learning_rate = engine_config.learning_rate,
                weight_decay = engine_config.weight_decay,
                warmup_ratio = engine_config.warmup_ratio,
                num_train_epochs = engine_config.nepochs,
                report_to = 'none',
                save_strategy = 'no'
        )
        self.trainer = Trainer(
                args = self.training_args,
                train_dataset = self.dataset,
                model = self.model
        )

    def __call__(self):
        self.trainer.train()

    @typechecked
    def get_train_test_split(self, config: ModelTrainingConfig) -> tuple[DataFrame , DataFrame| None]:
        data_coll = []
        for dataset in config.schemas:

            features = dataset.features.copy()

            if dataset.target not in features:
                features.append(dataset.target)

            if config.fold >= 0:
                features.append('fold')

            for file in dataset.train:
                data = load_csv(config.indir.path / dataset.name / file)

                if all([col in data.columns for col in features]):
                    data_coll.append(data[features])
                else:
                    logger.error(
                        f"The dataset can't be inlcuded as it have unmatched columns names {data.columns}"
                    )
        data = pd.concat(data_coll, axis=0)
        if config.fold >= 0 and config.fold <= data.fold.max():
            train_data = data.query("fold != @config.fold")
            valid_data = data.query("fold == @config.fold")
        else:
            train_data = data
            valid_data = None
        return train_data, valid_data

