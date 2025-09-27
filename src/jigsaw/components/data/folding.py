from pandas.core.frame import DataFrame
from ...entity.config_entity import DataTransformationConfig
from sklearn.model_selection import KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from ...utils.common import save_csv
from ...entity.common import Directory
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from ... import logger
from pathlib import Path


def split_dataset(
    config: DataTransformationConfig,
    data: DataFrame,
    path: list,
    name: str,
    outdir: Path | str | None = None,
) -> DataFrame:
    dataname, filename = path
    if filename == "sample_submission.csv":
        return data
    split_config = config.splitter

    data["fold"] = -1
    _splits_ = ["kfold", "skfold", "mlskfold"]

    if dataname in config.targets or split_config.labels:
        labels = config.targets[dataname]
        if split_config.labels:
            labels = split_config.labels

        if labels and _splits_.index("skfold") > _splits_.index(split_config.type):
            split_config.type = "skfold"

    if isinstance(labels, list):
        if len(labels) > 1:
            split_config.type = "mlskfold"
        else:
            labels = labels[0]
            split_config.type = "skfold"

    if split_config.type == "kfold":
        splitter = KFold
    elif split_config.type == "skfold":
        splitter = StratifiedKFold
    elif split_config.type == "mlskfold":
        splitter = MultilabelStratifiedKFold
        if hasattr(splitter, "labels"):
            labels = split_config.labels

    splitter = splitter(
        split_config.nsplits,
        shuffle=hasattr(split_config, "random_state"),
        random_state=split_config.random_state,
    )

    if labels:
        le_columns = []
        if isinstance(labels, list):
            for col in labels:
                if not is_integer_dtype(data[col]):
                    logger.info(f"Label Encoding the dataset column '{col}'")
                    data[f"{col}_le"] = LabelEncoder().fit_transform(data[col])
                    col = f"{col}_le"
                le_columns.append(col)

        elif isinstance(labels, str):
            le_columns = labels
            if not is_integer_dtype(data[labels]):
                le_columns = f"{labels}_le"
                data[le_columns] = LabelEncoder().fit_transform(data[labels])

        else:
            raise Exception("Labels are neither str or list[str]")

        logger.info(
            f"Folding '{dataname}.{filename}' into {split_config.nsplits} using {split_config.type} on column(s) {le_columns}"
        )

        for fold, (_, test_index) in enumerate(splitter.split(data, data[le_columns])):
            data.loc[test_index, "fold"] = fold

        if isinstance(le_columns, str):
            le_columns = le_columns if le_columns.endswith("_le") else None
        elif isinstance(le_columns, list):
            le_columns = [col for col in le_columns if col.endswith("_le")]

        if le_columns:
            data = data.drop(le_columns, axis=1)

    else:
        logger.error(
            f"Labels are not given for {dataname}' to use {split_config.type} Folding. Using Regular KFold"
        )

        splitter = KFold(
            split_config.nsplits,
            shuffle=hasattr(split_config, "random_state"),
            random_state=split_config.random_state,
        )

        for fold, (_, test_index) in enumerate(splitter.split(data)):
            data.loc[test_index, "fold"] = fold

    if outdir:
        target_dir = Directory(path=outdir / name)
        save_csv(data, target_dir.path / filename)

    return data
