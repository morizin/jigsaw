from .. import logger
from pydantic import validate_call
from pathlib import Path
from ..types import ZipFile, Directory
import zipfile
import os

@validate_call
def unzip_file(filepath : ZipFile, outdir : Directory):
    with zipfile.ZipFile(filepath.path, 'r') as zipfile:
        zipfile.extractall(path = outdir.path)

@validate_call
def create_directories(path: str) -> Directory:
    if isinstance(path, str):
        return Directory(path = path)

print(os.environ.get('KAGGLE_USERNAME'), os.environ.get('KAGGLE_KEY'))
