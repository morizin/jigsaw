from pydantic import BaseModel, field_validator
from typeguard import typechecked
from pathlib import Path
from typing import Union
from .. import logger
import os

FilePath = Union[str, os.PathLike]


class ZipFile(BaseModel):
    path: Path

    @field_validator("path", mode="before")
    @typechecked
    def is_zip_compatible(cls, v: Path | str) -> Path:
        if isinstance(v, str):
            v = Path(v)
        v = v.expanduser()
        if any(ext in [".zip", ".tar", ".gz"] for ext in v.suffixes):
            logger.info("Valid Zip File")
            return v
        return None


class Directory(BaseModel):
    path: Path

    @field_validator("path", mode="before")
    @typechecked
    def is_directory(cls, path: str | Path) -> Path:
        # Check if its a valid directory format
        if isinstance(path, str):
            path = Path(path)
        path = path.expanduser()
        Directory.create(path)
        return path

    @typechecked
    def __floordiv__(self, other: FilePath | "Directory") -> "Directory":
        if isinstance(other, FilePath):
            path = self.path / other
        if isinstance(other, Directory):
            path = self.path / other.path

        return Directory(path=path)

    @typechecked
    def __truediv__(self, other: FilePath | "Directory") -> FilePath:
        if isinstance(other, FilePath):
            path = self.path / other
        if isinstance(other, Directory):
            path = self.path / other.path
        return path

    @staticmethod
    def create(path: FilePath) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            logger.info(f"Creating Directory {str(path)}")
            path.mkdir(parents=True, exist_ok=True)

    def __str__(
        self,
    ):
        return self.path.as_posix()

    def listdir(self, *args: tuple[str]) -> list[FilePath]:
        path: FilePath = self.path
        if args is not None:
            for fold in args:
                path /= fold
                if not path.exists():
                    e = FileNotFoundError(path)
                    logger.error(e)
                    raise e
        return list(map(lambda x: os.path.basename(x.as_posix()), self.path.iterdir()))

    def exists(
        self,
    ) -> bool:
        return self.path.exists()
