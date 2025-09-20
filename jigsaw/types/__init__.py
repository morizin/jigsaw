from pydantic import BaseModel, field_validator, validate_call
from pathlib import Path
from typing import Any
from .. import logger

class ZipFile(BaseModel):
    path : Path

    @field_validator('path', mode = 'before')
    @validate_call
    def is_zip_compatible(cls , v: Path | str) -> Path:
        if isinstance(v, str):
            v = Path(v)
        v = v.expanduser()
        if any(ext in ['.zip', '.tar', '.gz'] for ext in v.suffixes):
             logger.info("Valid Zip File")
             return v
        return None

class Directory(BaseModel):
    path : Path

    @field_validator('path', mode= 'before')
    @validate_call
    def is_directory(cls, v : str | Path) -> Path:
       # Check if its a valid directory format 
        if isinstance(v, str):
            v = Path(v)
        v = v.expanduser()
        if not v.exists():
            logger.info(f"Creating Directory {v.as_posix()}")
            v.mkdir(parents = True, exist_ok = True)
        return v
