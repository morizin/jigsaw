from ....core.config_entity import DataTransformationConfig
from .... import logger
from ....errors import TransformationWarning, TransformationError
from ....core import Directory
from ....utils.common import save_csv
import pandas as pd
from typeguard import typechecked


@typechecked
def remove_duplicates(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    dataname: str,
    filename: str,
    outdir: Directory | None = None,
) -> pd.DataFrame:
    if filename != "sample_submission.csv":
        try:
            features = config.schemas[dataname].features
            if features:
                try:
                    data.drop_duplicates(
                        subset=features, ignore_index=True, inplace=True
                    )
                    logger.info(f"cleaning out duplicates: {dataname}.{filename}")
                except Exception as e:
                    e = TransformationError(
                        "Failed cleaning out duplicates",
                        dataname=dataname,
                        file_name=filename,
                        error=e,
                    )
                    logger.error(e)
                    raise e
        except Exception as e:
            w = TransformationWarning(
                "Failed cleaning out duplicates",
                dataname=dataname,
                file_name=filename,
                error=e,
            )
            logger.warning(w)
            data.drop_duplicates(ignore_index=True, inplace=True)

    if outdir:
        save_csv(data, outdir // f"cleaning_{dataname}" / filename)
    return data
