from pydantic import BaseModel
from jigsaw.types import Directory

class DataIngestionConfig(BaseModel):
    source : str
    type : str
    name : str
    outdir : Directory


