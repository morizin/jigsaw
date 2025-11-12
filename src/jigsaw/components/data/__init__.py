from sklearn.model_selection import StratifiedKFold
from .ingestion import DataIngestionComponent
from .validation import DataValidationComponent
from .transformation import DataTransformationComponent
from ...utils.common import load_csv
import pandas as pd
import os


def get_data(training_config, config, preprocess=None):
    test_data = []
    for schema in training_config.schemas:
        files = schema.test

        features = schema.features.copy()
        if schema.target not in features:
            features += [schema.target]

        for file in files:
            file = training_config.indir.path / schema.name / file
            if file.exists():
                data = load_csv(file)
                if all([feature in data.columns for feature in features]):
                    test_data.append(data[features])

    test_data = pd.concat(test_data, axis=0).reset_index(drop=True)
    if preprocess is not None:
        test_data = test_data.apply(preprocess, axis=1)

    if "out_of_fold" in config and config["out_of_fold"]["fold"] >= 0:
        mlskf = StratifiedKFold(
            n_splits=config["out_of_fold"]["n_splits"],
            shuffle=True,
            random_state=config["out_of_fold"]["seed"],
        )

        train_index, valid_index = list(
            mlskf.split(test_data, test_data["rule_violation"])
        )[config["out_of_fold"]["fold"]]
        valid_data, test_data = (
            test_data.iloc[valid_index, :],
            test_data.iloc[train_index, :].reset_index(drop=True),
        )
    else:
        valid_data = None

    train_data = []
    for schema in training_config.schemas:
        features = schema.features.copy()
        if schema.target not in features:
            features += [schema.target]

        for file in schema.train:
            file = training_config.indir.path / schema.name / file
            if file.exists():
                data = load_csv(file)
                if all([feature in data.columns for feature in features]):
                    train_data.append(data[features])

    train_data = pd.concat(train_data, axis=0).reset_index(drop=True)
    if preprocess is not None:
        train_data = train_data.apply(preprocess, axis=1)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN") is None:
        train_data = train_data[:225]
    return test_data, valid_data, train_data


__all__ = [
    "DataIngestionComponent",
    "get_data",
    "DataValidationComponent",
    "DataTransformationComponent",
]
