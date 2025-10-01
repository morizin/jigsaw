from ...entity.config_entity import DataTransformationConfig
from ... import logger
from ...entity.common import FilePath, Directory
from ...utils.common import save_csv
from pandas.core.frame import DataFrame
from pandas.api.types import is_string_dtype
from pathlib import Path
import pandas as pd
from cleantext import clean
import re
from typeguard import typechecked

@typechecked
def remove_duplicates(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    path: list[str],
    name: str,
    outdir: FilePath | None = None,
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

@typechecked
def clean_text(
    config: DataTransformationConfig,
    data: DataFrame,
    path: list,
    name: str,
    outdir: FilePath | None = None
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

@typechecked
def urlparse(
        config: DataTransformationConfig, 
        data: DataFrame,
        path : list[str],
        name : str,
        outdir: FilePath | None = None
        ) -> DataFrame:
    def url_to_semantics(text: str) -> str:
        if not isinstance(text, str):
            return ""

        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        urls = re.findall(url_pattern, text)
        
        if not urls:
            return "" 

        all_semantics = []
        seen_semantics = set()

        for url in urls:
            url_lower = url.lower()
            
            domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
            if domain_match:
                full_domain = domain_match.group(1)
                parts = full_domain.split('.')
                for part in parts:
                    if part and part not in seen_semantics and part != 'www': # Avoid short parts like 'www'
                        all_semantics.append(f"domain:{part}")
                        seen_semantics.add(part)

            # 2. Extract path parts
            path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
            path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters
            if path_parts:
                all_semantics.append("path:")
            for part in path_parts:
                # Clean up potential file extensions or query params
                part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
                if part_clean and part_clean not in seen_semantics:
                    all_semantics.append(f"{part_clean}")
                    seen_semantics.add(part_clean)

        if not all_semantics:
            return ""

        return f"\nURL Keywords: {' '.join(all_semantics)}"
    
    dataname, filename = path
    
    try:
        logger.info(f"Parsing URL of '{dataname}.{filename}'")
        for (col, dtype) in data.dtypes.items():
            if is_string_dtype(dtype):
                data[col] += data[col].apply(url_to_semantics)
    except Exception as e:
        logger.error("Parsing URL failed for f'{dataname}.{filename}': {e}")
        raise e

    if outdir:
        target_dir = Directory(path = (Path(outdir) if isinstance(outdir, str) else outdir) / name)
        save_csv(data, target_dir.path / filename)
    return data
