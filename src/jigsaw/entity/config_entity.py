from pydantic import BaseModel
from .common import Directory

class DataSource(BaseModel):
    source : str
    type : str
    name : str

class DataIngestionConfig(BaseModel):
    sources : list[DataSource]
    names : list[str]
    outdir : Directory

