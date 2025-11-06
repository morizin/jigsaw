from pandas.core.frame import DataFrame
from ...schema.config_entity import DataTransformationConfig
from ....utils.common import save_csv
from ...schema.config_entity import Directory
from .cleaning import remove_duplicates
from typeguard import typechecked
from .... import logger
from pathlib import Path
import pandas as pd


@typechecked
def zero_shot_transform(
    config: DataTransformationConfig,
    data: DataFrame,
    path: list,
    name: str,
    outdir: Path | str | None = None,
) -> DataFrame:
    dataname, filename = path
    try:
        if filename != "sample_submission.csv":
            features = config.features[dataname]
            if not features:
                logger.error(
                    f"Error Transforming to Zero-shot Dataset: '{dataname}' doesn't have features indicated"
                )
                raise Exception(
                    f"Error Transforming to Zero-shot Dataset: '{dataname}' doesn't have features indicated"
                )

            try:
                zeroshot = [data[features + ["rule_violation"]]]
            except KeyError:
                zeroshot = []
            except Exception as e:
                raise e

            for violation in ["positive", "negative"]:
                for i in range(1, 3):
                    temp = data[features[:-1] + [f"{violation}_example_{i}"]]
                    temp["rule_violation"] = 1 if violation == "positive" else 0
                    temp = temp.rename(columns={f"{violation}_example_{i}": "body"})
                    zeroshot.append(temp)

            zeroshot = pd.concat(zeroshot, axis=0)

            logger.info(f"Tranforming to Zero-Shot Dataset : {dataname}.{filename}")
        else:
            zeroshot = data

        zeroshot = remove_duplicates(config, zeroshot, path, name="")

    except Exception as e:
        logger.error(f"Error Tranforming to Zero-Shot Dataset : {dataname}.{filename}")
        raise e

    if outdir:
        target_dir = Directory(path=outdir / name)
        save_csv(zeroshot, target_dir.path / filename)
    return zeroshot
