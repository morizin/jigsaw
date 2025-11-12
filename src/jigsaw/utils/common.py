from ..core.io_types import ZipFile, Directory, FilePath
from .. import logger
from .yaml_loader import YAMLoader
from typeguard import typechecked
from box import ConfigBox
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import subprocess
import platform
import zipfile
import random
import os
import re
import pickle
import torch
import yaml
import json


def get_os_type():
    os_name = platform.system()
    if os_name == "Windows":
        return "Windows"
    elif os_name == "Darwin":  # Darwin is the kernel name for macOS
        return "macOS"
    elif os_name == "Linux":
        return "Linux"
    else:
        return f"Unknown OS: {os_name}"


def get_hw_details():
    os_name = get_os_type()
    match os_name:
        case "Windows":
            n_procs = subprocess.check_output("", shell=True).decode()
            n_threads = subprocess.check_output("", shell=True).decode()
        case "macOS":
            n_procs = subprocess.check_output(
                "sysctl -n hw.physicalcpu", shell=True
            ).decode()
            n_threads = subprocess.check_output(
                "sysctl -n hw.logicalcpu", shell=True
            ).decode()
        case "Linux":
            n_procs = subprocess.check_output(
                'lscpu | grep -P ^Thread\(s\) per core\:.*(\d+).* | grep -oE "\d+"',
                shell=True,
            ).decode()
            n_threads = subprocess.check_output(
                'lscpu | grep -P ^Core\(s\) per socket\:.*(\d+).* | grep -oE "\d+"',
                shell=True,
            ).decode()
        case _:
            n_procs = 1
            n_threads = 1

    if isinstance(n_procs, str):
        n_procs = int(n_procs.strip())

    if isinstance(n_threads, str):
        n_threads = int(n_threads.strip())

    if os_name == "macOS":
        n_threads //= n_procs

    return n_procs, n_threads


@typechecked
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


@typechecked
def unzip_file(filepath: ZipFile, outdir: Directory):
    logger.info(f"Extracting {filepath.path} to {outdir.path}")
    try:
        with zipfile.ZipFile(filepath.path, "r") as file:
            file.extractall(path=outdir.path)
    except FileNotFoundError as e:
        logger.error(e)
        raise e
    except Exception as e:
        logger.error(f"Error Loading file : {str(filepath.path)}")
        raise e


@typechecked
def save_pickle(data: Any, path: FilePath, verbose=int(os.getenv("VERBOSITY"))) -> Any:
    try:
        data = pickle.dump(data, open(path, "wb"))
        if verbose:
            logger.info(f"Successfully Saving the file : {str(path)}")
        return data
    except Exception as e:
        logger.error(f"Error Saving Binary file : {str(path)}")
        raise e


@typechecked
def load_pickle(path: FilePath, verbose=int(os.getenv("VERBOSITY"))) -> Any:
    try:
        data = pickle.load(open(path, "rb"))
        if verbose:
            logger.info(f"Successfully loaded the file : {str(path)}")
        return data

    except FileNotFoundError as e:
        logger.error(e)
        raise e
    except Exception as e:
        logger.error(f"Error Loading file : {str(path)}")
        raise e


@typechecked
def save_yaml(data: Any, path: FilePath, verbose=int(os.getenv("VERBOSITY"))):
    try:
        path = Directory(path=os.path.dirname(path)) / os.path.basename(path)
        path = str(path)
        if not path.endswith(".yaml") or path.endswith(".yml"):
            path = re.sub("\.[^.]+$", "", path) + ".yaml"
        content = yaml.dump(data, open(path, "w"))
        if verbose:
            logger.info(
                f"Created Yaml file at {str(path)} with {len(data)} master keys."
            )
    except Exception as e:
        e = f"Error Loading YAML : {path} : {e}"
        logger.error(e)
        raise (e)
    return content


@typechecked
def load_yaml(
    path: FilePath, box=True, verbose=int(os.getenv("VERBOSITY"))
) -> dict | ConfigBox:
    if isinstance(path, Path):
        path = path.as_posix()

    try:
        content = yaml.load(open(path), Loader=YAMLoader)
        if box:
            content = ConfigBox(content)
        if verbose:
            logger.info(f"Successfully loaded the YAML file : {path}")
    except FileNotFoundError as e:
        logger.error(e)
        raise e
    except Exception as e:
        e = f"Error Loading YAML : {path} : {e}"
        logger.error(e)
        raise (e)
    return content


@typechecked
def save_csv(
    data: pd.DataFrame,
    path: str | Path,
    verbose=int(os.getenv("VERBOSITY")),
):
    try:
        Directory(path=os.path.split(path)[0])
        data.to_csv(path, index=False)
        if verbose:
            logger.info(f"Saved the data into {str(path)}")
    except Exception as e:
        logger.error(f"Error when Saving {str(path)} : {e}")
        raise e


@typechecked
def load_csv(
    path: FilePath,
    verbose=int(os.getenv("VERBOSITY")),
) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        if verbose:
            logger.info(f"Successfully read the CSV {str(path)} : {data.shape}")
        return data
    except FileNotFoundError as e:
        logger.error(e)
        raise e
    except Exception as e:
        logger.exception(f"Error Reading File: {str(path)}")
        raise e


@typechecked
def save_json(
    data: Any,
    path: FilePath,
    verbose=int(os.getenv("VERBOSITY")),
):
    try:
        path = Directory(path=os.path.dirname(path)) / os.path.basename(path)
        path = str(path)
        if not path.endswith(".json"):
            path = re.sub("\.[^.]+$", "", path) + ".json"
        json.dump(data, open(path, "w"), indent=2)
        if verbose:
            logger.info(f"jsonified data at {str(path)} -> {len(data)} master keys.")
    except Exception as e:
        logger.error(f"Error when jsonifying {str(path)} : {e}")
        raise e


@typechecked
def load_json(
    path: FilePath,
    verbose=int(os.getenv("VERBOSITY")),
) -> dict:
    try:
        data = json.load(open(path))
        if verbose:
            logger.info(
                f"Successfully loaded file: {str(path)} -> {len(data)} master keys."
            )
        return data
    except FileNotFoundError as e:
        logger.error(e)
        raise e
    except Exception as e:
        logger.exception(f"Error when loading file : {str(path)}")
        raise e
