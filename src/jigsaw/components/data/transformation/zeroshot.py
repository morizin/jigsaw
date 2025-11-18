from pandas.core.frame import DataFrame
from ....core.config_entity import DataTransformationConfig, Directory
from ....errors import TransformationError
from ....utils.common import save_csv
from .cleaning import remove_duplicates
from typeguard import typechecked
from .... import logger
import pandas as pd


@typechecked
def zero_shot_transform(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    dataname: str,
    filename: str,
    outdir: Directory | None = None,
) -> DataFrame:
    schema = config.schemas[dataname]
    try:
        features = schema.features.copy()
        if not features:
            e = TransformationError(
                "Error Transforming to Zero-shot Dataset: doesn't have features indicated",
                dataname=dataname,
                file_name=filename,
            )
            logger.error(e)
            raise e

        zeroshot = []
        if schema.target in data.columns:
            features += (
                [schema.target] if isinstance(schema.target, str) else schema.target
            )

        zeroshot.append(data[features])

        for violation in ["positive", "negative"]:
            for i in range(1, 3):
                temp = data[["rule", f"{violation}_example_{i}"]]
                temp["rule_violation"] = 1 if violation == "positive" else 0
                temp = temp.rename(columns={f"{violation}_example_{i}": "body"})
                zeroshot.append(temp)

        zeroshot = pd.concat(zeroshot, axis=0)

        logger.info(f"Tranforming to Zero-Shot Dataset : {dataname}.{filename}")
        zeroshot = remove_duplicates(config, zeroshot, dataname, filename)

    except Exception as e:
        e = TransformationError(
            "Error Tranforming to Zero-Shot Dataset",
            dataname=dataname,
            file_name=filename,
            error=e,
        )
        logger.error(e)
        zeroshot = data.copy()

    if outdir:
        save_csv(zeroshot, outdir // f"zero_shot_{dataname}" / filename)
    return zeroshot
