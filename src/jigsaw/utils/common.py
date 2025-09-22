from .. import logger
from pydantic import validate_call
from ..entity.common import ZipFile, Directory
from pathlib import Path
import zipfile
import os
import yaml
from box import ConfigBox

@validate_call
def unzip_file(filepath : ZipFile, outdir : Directory):
    logger.info(f"Extracting {filepath.path} to {outdir.path}")
    with zipfile.ZipFile(filepath.path, 'r') as file:
        file.extractall(path = outdir.path)

@validate_call
def load_yaml(filepath : Path | str) -> ConfigBox:
    if isinstance(filepath, Path):
        filepath = filepath.as_posix()
    try:
        logger.info(f"Loading YAML file : {filepath}")
        content = yaml.safe_load(open(filepath))
        content = ConfigBox(content)
    except Exception as e:
        logger.error(f"Error Loading YAML : {filepath} : {e}")
    return content 
