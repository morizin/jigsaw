from pydantic import validate_call
from src.jigsaw import logger
from src.jigsaw.entity.common import ZipFile, Directory, FilePath
from pathlib import Path
import os, zipfile, yaml, json
from box import ConfigBox
import pandas as pd
from typing import Any
from collections import defaultdict


@validate_call
def unzip_file(filepath: ZipFile, outdir: Directory):
    logger.info(f"Extracting {filepath.path} to {outdir.path}")
    with zipfile.ZipFile(filepath.path, "r") as file:
        file.extractall(path=outdir.path)


@validate_call
def load_yaml(filepath: Path | str) -> ConfigBox:
    if isinstance(filepath, Path):
        filepath = filepath.as_posix()
    try:
        logger.info(f"Loading YAML file : {filepath}")
        content = yaml.safe_load(open(filepath))
        content = ConfigBox(content)
    except Exception as e:
        logger.error(f"Error Loading YAML : {filepath} : {e}")
    return content


@validate_call
def read_csv(path: FilePath) -> pd.DataFrame:
    try:
        logger.info(f"Reading {path} file ...")
        data = pd.read_csv(path)
        logger.info(f"Successfully read the CSV {path}")
        return data
    except Exception as e:
        logger.exception(f"Error Reading File: {path}")
        raise e


@validate_call
def save_json(data: Any, path: FilePath):
    try:
        logger.info(f"jsonifying data to {path} ... ")
        Directory(path=os.path.split(path)[0])
        json.dump(data, open(path, "w"))
        logger.info(f"jsonified data at {path}")
    except Exception as e:
        logger.error(f"Error when jsonifying {path} : {e}")
        raise e
