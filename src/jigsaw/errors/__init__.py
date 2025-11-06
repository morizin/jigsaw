from ..core import Directory, FilePath
from ..components import Component
from typing import Callable
import os


class PipelineError(Exception):
    def __init__(self, stage: str, error: str = ""):
        super().__init__(f"Error occured at stage {stage} : {error}".strip())


class ComponentError(Exception):
    def __init__(self, object: Component | str, error: str = ""):
        if isinstance(object, Component):
            object = object.__class__.__name__
        super().__init__(f"Error occured at component {object} : {error}".strip())


class DataNotFoundError(Exception):
    def __init__(self, data_path: str, error: str = ""):
        super().__init__(
            f"Unable to find data '{os.path.basename(data_path)}' at {os.path.dirname(data_path)} {error}".strip()
        )


class DirectoryNotFoundError(Exception):
    def __init__(self, path: FilePath | Directory, error: str = ""):
        if isinstance(path, Directory):
            path = path.path

        message = f"Unable to find directory {path}"
        if error:
            message += f" : {error}"
        super().__init__(message)


class ValidationError(Exception):
    def __init__(self, object=None, message: str = ""):
        if object is not None:
            message += f" : {object}"
        super().__init__(message)


class ValidationWarning(Warning):
    def __init__(self, object=None, message: str = ""):
        if object is not None:
            message += f" : {object}"
        super().__init__(message)
