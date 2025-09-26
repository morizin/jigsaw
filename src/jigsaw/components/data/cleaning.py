from src.jigsaw.entity.config_entity import DataTransformationConfig
from src.jigsaw import logger
from src.jigsaw.entity.common import Directory
from src.jigsaw.utils.common import save_csv
from pandas.core.frame import DataFrame
from pandas.api.types import is_string_dtype
from pathlib import Path
import pandas as pd
from cleantext import clean
from ensure import ensure_annotations

@ensure_annotations
def remove_duplicates(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    path: list,
    name: str,
    outdir: Path | str | None = None,
) -> pd.DataFrame:
    dataname, filename = path

    if filename != "sample_submission.csv":
        features = config.features[dataname]
        if features:
            try:
                data.drop_duplicates(subset=features, ignore_index=True, inplace=True)
                logger.info(f"cleaning out duplicates: {dataname}.{filename}")
            except Exception:
                logger.error(
                    f"Failed cleaning out duplicates: {dataname}.{filename}\nManual Cleaning"
                )
                data.drop_duplicates(ignore_index=True, inplace=True)
        else:
            data.drop_duplicates(ignore_index=True, inplace=True)

    if outdir:
        target_dir = Directory(path=Path(outdir) / name)
        save_csv(data, target_dir.path / filename)
    return data

from pandas.core.frame import DataFrame
from src.jigsaw.entity.common import Directory
from src.jigsaw.utils.common import save_csv
from src.jigsaw import logger
from pandas.api.types import is_string_dtype

@ensure_annotations
def clean_text(
    config: DataTransformationConfig,
    data: DataFrame,
    path: list,
    name: str,
    outdir: Path | str | None = None
) -> DataFrame:
    def clean_text(text):
        return clean(
            text,
            fix_unicode=True,
            to_ascii=True,
            lower=False,
            no_line_breaks=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            no_emoji=True,
            replace_with_url="<URL>",
            replace_with_phone_number="<PHONE>",
            replace_with_email="<EMAIL>",
        )

    dataname, filename = path

    if filename != "sample_submission.csv":
        for key, dtype in data.dtypes.items():
            if is_string_dtype(dtype):
                data[key] = data[key].apply(clean_text)
        logger.info(f"Washed the file : {dataname}.{filename}")
        remove_duplicates(config, data, path, name="")
    else:
        logger.warning(f"Couldn't clean text in {dataname}.{filename}")

    if outdir:
        target_dir = Directory(path=outdir / name)
        save_csv(data, target_dir.path / filename)
    return data
