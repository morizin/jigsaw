from .data.augmentation import Augmentor
from ..utils.common import load_csv, load_json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from .. import PROJECT_NAME

from ..core import (
    FilePath,
    ModelTrainingConfig,
    ModelTrainingArtifact,
    ClassificationMetric,
    DataTransformationArtifact,
)

from .data.transformation import DataTransformationComponent

from transformers.utils import is_torch_bf16_gpu_available
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
from scipy.special import softmax as softmax_scipy
import numpy as np


class ModelTrainingComponent:
    def __init__(self, config: FilePath | ModelTrainingConfig):
        if isinstance(config, FilePath):
            config = ModelTrainingConfig(**load_json(config))

        self.config = config
        self.transform_config = config.transformation
        self.transform_artifact: DataTransformationArtifact = (
            DataTransformationComponent(self.transform_config)()
        )
        print(self.transform_artifact)
        self.exp_name = f"{PROJECT_NAME}_{config.name.replace('/', '_')}"

    def get_dataset(self, valid=False):
        file_path = (
            self.transform_artifact.valid_file_path
            if valid
            else self.transform_artifact.train_file_path
        )

        data = load_csv(file_path)
        if self.transform_config.augmentations:
            augs = Augmentor(
                augments=[["url_to_semantics", 1.0]],
                frac=1.0,
                resample=1,
                include_original=False,
                weight=1,
            )
            data = augs.augment(data)

        dataset = JigsawDataset(self.config, data, self.tokenizer)
        return dataset

    def get_model_tokenizer(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model, num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

    def get_trainer(self):
        self.get_model_tokenizer()
        train_dataset = self.get_dataset()

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=str(self.config.outdir),
            per_device_train_batch_size=self.config.train_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            num_train_epochs=self.config.n_epochs,
            lr_scheduler_type=self.config.scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            do_eval=self.config.do_eval,
            per_device_eval_batch_size=self.config.valid_batch_size,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_dir=self.config.logging_dir,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_safetensors=True,
            seed=self.config.seed,
            bf16=is_torch_bf16_gpu_available(),
            fp16=not is_torch_bf16_gpu_available(),
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if self.config.gradient_checkpointing
            else None,
            report_to=["mlflow"],
            optim=self.config.optimizer,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )

    def __call__(
        self,
    ) -> ModelTrainingArtifact:
        mlflow.set_experiment(self.exp_name)
        self.trainer = self.get_trainer()
        include_fields = {
            "name",
            "type",
            "augmentations",
            "train_file_path",
            "valid_file_path",
            "max_length",
            "padding",
            "fold",
            "out_of_fold",
        }

        with mlflow.start_run():
            params = self.config.to_mlflow_params(include=include_fields)
            mlflow.log_params(params)
            self.train()

            metrics: ClassificationMetric = self.evaluate()
            mlflow.log_metrics(metrics.model_dump())

        return ModelTrainingArtifact(
            name=self.config.name,
            model_path=self.config.outdir,
            metrics=metrics,
            # metrics=ClassificationMetric(roc_auc=0.5, accuracy=0.5),
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(str(self.config.outdir))

    def evaluate(self) -> ClassificationMetric:
        test_dataset = self.get_dataset(valid=True)
        predictions = self.trainer.predict(test_dataset)
        preds, labels = (
            softmax_scipy(predictions.predictions, axis=1)[:, 1],
            test_dataset.labels,
        )
        roc_auc = roc_auc_score(labels, preds)
        accuracy = accuracy_score(labels, (preds > 0.5).astype(np.int32))
        return ClassificationMetric(roc_auc=roc_auc, accuracy=accuracy)


class JigsawDataset:
    def __init__(self, config, data, tokenizer):
        data["prompt"] = data["rule"] + "[SEP]" + data["body"]
        self.data = data
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        self.encodings = tokenizer(
            data["prompt"].tolist(), truncation=True, max_length=config.max_length
        )
        self.labels = (
            self.data["rule_violation"].tolist()
            if "rule_violation" in self.data.columns
            else None
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(
        self,
        idx: int,
    ):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item
