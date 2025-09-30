from .. import logger
from ..entity.common import ZipFile, Directory, FilePath
from .yaml_loader import YAMLoader
from pathlib import Path
import os, zipfile, yaml, json, pickle
from box import ConfigBox
import pandas as pd
from typing import Any
from collections import defaultdict
from typeguard import typechecked

@typechecked
def unzip_file(filepath: ZipFile, outdir: Directory):
    logger.info(f"Extracting {filepath.path} to {outdir.path}")
    with zipfile.ZipFile(filepath.path, "r") as file:
        file.extractall(path=outdir.path)


@typechecked
def load_yaml(filepath: FilePath) -> ConfigBox:
    if isinstance(filepath, Path):
        filepath = filepath.as_posix()
    try:
        logger.info(f"Loading YAML file : {filepath}")
        content = yaml.load(open(filepath), Loader=YAMLoader)
        #         content = yaml.safe_load(open(filepath))
        content = ConfigBox(content)
    except Exception as e:
        logger.error(f"Error Loading YAML : {filepath} : {e}")
    return content

@typechecked
def load_csv(path: FilePath) -> pd.DataFrame:
    try:
        logger.info(f"Reading {str(path)} file ...")
        data = pd.read_csv(path)
        logger.info(f"Successfully read the CSV {str(path)} : {data.shape}")
        return data
    except Exception as e:
        logger.exception(f"Error Reading File: {str(path)}")
        raise e


@typechecked
def save_json(data: Any, path: FilePath):
    try:
        logger.info(f"jsonifying data to {str(path)} ... ")
        Directory(path=os.path.split(path)[0])
        json.dump(data, open(path, "w"))
        logger.info(f"jsonified data at {str(path)} -> {len(data)} master keys.")
    except Exception as e:
        logger.error(f"Error when jsonifying {str(path)} : {e}")
        raise e


@typechecked
def load_json(path: FilePath) -> dict:
    try:
        logger.info(f"Loading Json file : {str(path)}")
        data = json.load(open(path))
        logger.info(
            f"Successfully loaded file: {str(path)} -> {len(data)} master keys."
        )
        return data
    except Exception as e:
        logger.exception(f"Error when loading file : {str(path)}")
        raise e


@typechecked
def save_csv(data: pd.DataFrame, path: str | Path):
    try:
        logger.info(f"Saving dataframe of {data.shape} into {str(path)}. ...")
        Directory(path=os.path.split(path)[0])
        data.to_csv(path, index=False)
        logger.info(f"Saved the data into {str(path)}")

    except Exception as e:
        logger.error(f"Error when Saving {str(path)} : {e}")
        raise e


@typechecked
def load_pickle(path: FilePath) -> Any:
    try:
        logger.info(f"Loading binary file : {str(path)}")
        data = pickle.load(open(path, "wb"))
        logger.info(f"Successfully loaded the file : {str(path)}")
        return data
    except Exception as e:
        logger.error(f"Error Loading file : {str(path)}")
        raise e


@typechecked
def print_format(string: FilePath, length: int):
    space = length - len(str(string)) - 1
    print(
        "|{}{}{}|".format(
            " " * (space // 2),
            string,
            " " * (space // 2),
        )
    )
