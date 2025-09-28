from pydantic import validate_call
import pandas as pd
from ... import logger
from ...entity.common import Directory
from ...entity.config_entity import DataTransformationConfig, DataSplitParams

from .cleaning import remove_duplicates, clean_text
from .zeroshot import zero_shot_transform
from .folding import split_dataset
from .triplet import triplet_dataset
from pathlib import Path
from ensure import ensure_annotations
from cleantext import clean
from pandas.api.types import is_string_dtype
from ...utils.common import read_csv, save_csv, print_format

class DataTransformationComponent:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        self.outdir = self.config.outdir.path
        self.indir = self.config.indir.path

        self.names = []
        self.pipeline = []

        final_dir = ""

        length = 100
        print("=" * length)
        print_format("Datasets Available", length)
        print("=" * length)
        for name in self.config.datasets:
            if (self.outdir / name).is_dir():
                print_format(name, length)
                self.names.append(str(name))
        print("=" * length)

        print()

        print("=" * length)
        print_format("Datasets Generating", length)
        print("=" * length)
        if self.config.features:
            for name in self.names:
                final_dir = "cleaned_" + final_dir
                self.pipeline.append((final_dir, remove_duplicates))
                print_format(self.indir / f"{final_dir}{name}/", length)
            print("=" * length)

        if self.config.wash:
            for name in self.names:
                final_dir = "washed_" + final_dir
                self.pipeline.append((final_dir, clean_text))
                print_format(self.indir / f"{final_dir}{name}/", length)
            print("=" * length)

        if self.config.zero:
            for name in self.names:
                final_dir = "zero_" + final_dir
                self.pipeline.append((final_dir, zero_shot_transform))
                print_format(self.indir / f"{final_dir}{name}/", length)
            print("=" * length)

        if self.config.triplet:
            for name in self.names:
                final_dir = "triplet_" + final_dir
                self.pipeline.append((final_dir, triplet_dataset))
                print_format(self.indir / f"{final_dir}{name}/", length)
            print("=" * length)

        if self.config.pairwise:
            for name in self.names:
                final_dir = "pairwise_" + final_dir
                print_format(self.indir / f"{final_dir}{name}/", length)
            print("=" * length)

        if self.config.splitter:
            for name in self.names:
                final_dir = "folded_" + final_dir
                self.pipeline.append((final_dir, split_dataset))
                print_format(self.indir / f"{final_dir}{name}/", length)

            print("=" * length)

        self.final_dir = final_dir

    @validate_call
    def __call__(self):
        for name in self.names:
            for path in (self.indir / name).iterdir():
                data = read_csv(path)
                path = str(path).split("/")[-2:]
                for (dirname, process) in self.pipeline:
                    data = process(
                        config= self.config,
                        data = data,
                        path = path,
                        name = dirname + name,
                        outdir = self.outdir
                    )
