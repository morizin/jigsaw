from pandas.api.types import is_object_dtype, is_integer_dtype, is_string_dtype
from ....constants.data import (
    VALID_PREFIX,
    INVALID_PREFIX,
    REPORT_IMAGES_DIR,
    REPORT_NAME,
)
from ....core import FilePath, DataValidationConfig, DataSchema, DataValidationArtifact
from ....utils.common import load_csv, save_json, save_csv
from ....errors import ValidationError, ValidationWarning
from ....constants import DATA_DIRECTORY_NAME
from collections import defaultdict
from typeguard import typechecked
from ....constants import LENGTH
from ... import Component
from .... import logger
import pandas as pd
import os


class DataValidationComponent(Component):
    @typechecked
    def __init__(self, data_validation_config: DataValidationConfig):
        self.config = data_validation_config
        self.names = list(self.config.schemas.keys())

        tree = lambda: defaultdict(tree)
        self.status = tree()

        length = max(LENGTH, max(list(map(len, self.names))))
        print("=" * length)
        print("|", "Datasets Available".center(length - 4), "|")
        print("=" * length)
        for name in self.names:
            print("|", name.center(length - 4), "|")
        print("=" * length)

        self.pipeline = [
            self.find_missing_columns,
            self.find_mismatch_dtype,
            self.find_data_redundancy,
            self.find_missing_rows,
        ]
        if self.config.statistics:
            self.pipeline.append(self.get_statistics)

        self.validation_status = True
        self.valid_outdir = (
            self.config.outdir // f"{VALID_PREFIX}_{DATA_DIRECTORY_NAME}"
        )
        self.invalid_outdir = (
            self.config.outdir // f"{INVALID_PREFIX}_{DATA_DIRECTORY_NAME}"
        )
        self.report = self.config.outdir // f"{self.config.report_name}"
        self.report_img = self.report // REPORT_IMAGES_DIR

    def __call__(self):
        for name, schema in self.config.schemas.items():
            for file in filter(lambda x: str(x).endswith(".csv"), schema.train):
                valid_data = load_csv(self.config.indir / name / file)
                for process in self.pipeline:
                    valid_data, invalid_data = process(
                        valid_data, name, file, schema, train=True
                    )
                    if valid_data is not None:
                        save_csv(valid_data, self.valid_outdir // name / file)
                    if invalid_data is not None:
                        save_csv(invalid_data, self.invalid_outdir // name / file)

            for file in filter(lambda x: str(x).endswith(".csv"), schema.test):
                valid_data = load_csv(self.config.indir / name / file)
                for process in self.pipeline:
                    valid_data, invalid_data = process(
                        valid_data, name, file, schema, train=False
                    )
                    if valid_data is not None:
                        save_csv(valid_data, self.valid_outdir // name / file)
                    if invalid_data is not None:
                        save_csv(invalid_data, self.invalid_outdir // name / file)

            for file in filter(lambda x: str(x).endswith(".csv"), schema.submission):
                submission_df = load_csv(self.config.indir / name / file)
                test_df = load_csv(
                    self.valid_outdir / name / ".".join(["test"] + file.split(".")[1:])
                )
                if test_df.shape[0] == submission_df.shape[0]:
                    save_csv(submission_df, self.valid_outdir // name / file)
                else:
                    save_csv(submission_df, self.invalid_outdir / name / file)

        self.status["validation_status"] = self.validation_status
        save_json(self.status, self.report / f"{REPORT_NAME}.json")
        return DataValidationArtifact(
            validation_status=self.validation_status,
            valid_outdir=self.valid_outdir,
            invalid_outdir=self.invalid_outdir,
            report_dir=self.report,
            schemas=self.config.schemas,
        )

    @typechecked
    def find_missing_columns(
        self,
        data: pd.DataFrame,
        name: FilePath,
        file: FilePath,
        schema: DataSchema,
        train=True,
    ):
        # data = load_csv(self.config.indir / name / file)
        data_cols = data.columns
        if not train:
            schema.schema.pop(schema.target)
        target_cols = schema.schema.keys()

        missing_col: list[str] = []
        for col in target_cols:
            if col not in data_cols:
                missing_col.append(col)

        self.status[str(name)][str(file)]["missing_columns"] = (
            missing_col if missing_col else False
        )

        if missing_col:
            self.validation_status = False
            e = ValidationError(
                missing_col, f"Found Missing Columns at '{name}.{file}'"
            )
            logger.error(e)
            raise e

        return data, None

    @typechecked
    def find_mismatch_dtype(
        self,
        data: pd.DataFrame,
        name: FilePath,
        file: FilePath,
        schema: DataSchema,
        train=True,
    ):
        # data = load_csv(self.config.indir / name / file)

        dtype_mismatch: list[dict[str, bool]] = []
        for column, dtype in schema.schema.items():
            if dtype.lower() in ("str", "string", "object"):
                if not is_object_dtype(data[column]):
                    dtype_mismatch.append(
                        {
                            "column": column,
                            "dtype": dtype,
                            "found": data[column].dtype.__class__.__name__.split(".")[
                                -1
                            ],
                        }
                    )

            if dtype.lower() in ("int", "int32", "int64", "integer", "natural"):
                if not is_integer_dtype(data[column]):
                    dtype_mismatch.append(
                        {
                            "column": column,
                            "dtype": dtype,
                            "found": data[column].dtype.__class__.__name__.split(".")[
                                -1
                            ],
                        }
                    )

        self.status[str(name)][str(file)]["dtype_mismatch"] = (
            dtype_mismatch if dtype_mismatch else False
        )
        mismatch_col = [i["column"] for i in dtype_mismatch]

        if mismatch_col:
            self.validation_status = False
            e = ValidationError(message=f"Invalid File {name}.{file}")
            logger.error(e)
            raise e

        return data.drop(mismatch_col, axis=1), data[mismatch_col]

    @typechecked
    def find_data_redundancy(
        self,
        data: pd.DataFrame,
        name: FilePath,
        file: FilePath,
        schema: DataSchema,
        **kwargs,
    ):
        try:
            # data = load_csv(self.config.indir / name / file)
            data_redundancy: dict[str, int | float] = {}
            data_redundancy["raw_count"] = data.shape[0]

            duplicates = data[data.duplicated(subset=schema.features)].index.tolist()
            data_redundancy["n_duplicates"] = len(duplicates)
            if data_redundancy["n_duplicates"] > 0:
                w = ValidationWarning(
                    f"Data Redundancy: {data_redundancy['n_duplicates']} duplicates has been found in {name}.{file}"
                )
                logger.warning(w)
                data_redundancy["purity"] = round(
                    100
                    * (data_redundancy["raw_count"] - data_redundancy["n_duplicates"])
                    / data_redundancy["raw_count"],
                    4,
                )
            else:
                data_redundancy["purity"] = 100
        except Exception as e:
            self.validation_status = False
            e = ValidationError(message=e)
            logger.error(e)
            raise e

        self.status[str(name)][str(file)]["data_redundancy"] = data_redundancy
        return data.drop(duplicates, axis=0), data.iloc[duplicates, :]

    @typechecked
    def find_missing_rows(
        self,
        data: pd.DataFrame,
        name: FilePath,
        file: FilePath,
        schema: DataSchema,
        **kwargs,
    ):
        try:
            # data = load_csv(self.config.indir / name / file)
            missingrows: dict[str, int | float] = {}
            missingrows["raw_count"] = data.shape[0]

            missing_rows = data[data[schema.features].isna().any(axis=1)].index.tolist()
            missingrows["n_missing"] = len(missing_rows)
            if missingrows["n_missing"] > 0:
                w = ValidationWarning(
                    f"Data Missing: {missingrows['n_missing']} rows has been found with NaN in {name}.{file}"
                )
                logger.warning(w)
                missingrows["purity"] = 100 * round(
                    (missingrows["raw_count"] - missingrows["n_missing"])
                    / missingrows["raw_count"],
                    4,
                )
            else:
                missingrows["purity"] = 100
        except Exception as e:
            self.validation_status = False
            e = ValidationError(message=e)
            logger.error(e)
            raise e

        self.status[str(name)][str(file)]["missing_rows"] = missingrows

        return data.drop(missing_rows, axis=0), data.iloc[missing_rows, :]

    @typechecked
    def get_statistics(
        self,
        data: pd.DataFrame,
        name: FilePath,
        file: FilePath,
        schema: DataSchema,
        train=True,
        **kwargs,
    ):
        from .text import (
            get_statistics as text_statistics,
            generate_word_cloud,
            detect_data_drift,
        )

        try:
            # data = load_csv(path=self.config.indir / name / file)
            statistics: dict[str, str | dict[str, int | float]] = dict()
            if train and isinstance(schema.target, str):
                target = [schema.target]
            else:
                target = []
                train_data = load_csv(
                    self.config.indir / name / file.replace("test", "train")
                )

            for column in data.columns:
                if column in schema.categorical:
                    statistics[column] = {"type": "categorical"}
                    statistics[column]["n_classes"] = data[column].nunique()
                    if statistics[column]["n_classes"] <= 10:
                        statistics[column]["classes"] = data[column].unique().tolist()
                        statistics[column]["class_distribution"] = (
                            data[column].value_counts().to_dict()
                        )
                        target_dist: dict[str, dict] = dict()
                        if column not in target:
                            for tcolumn in target:
                                if tcolumn in schema.categorical:
                                    target_dist[tcolumn] = (
                                        data.groupby(by=column)[tcolumn]
                                        .value_counts()
                                        .reset_index()
                                        .to_dict(orient="records")
                                    )

                        statistics[column]["target_dist"] = (
                            target_dist if target_dist else None
                        )
                    else:
                        statistics[column]["classes"] = None
                        statistics[column]["class_distribution"] = None
                        statistics[column]["target_dist"] = None

                elif is_string_dtype(data[column]):
                    statistics[column] = text_statistics(
                        data[column],
                        column=column,
                        path=self.report_img // name // file,
                    )
                    statistics[column]["word_cloud"] = generate_word_cloud(
                        data=data,
                        column=column,
                        path=self.report_img // name // file,
                        seed=int(os.environ["PYTHONHASHSEED"]),
                    )
                    if self.config.data_drift:
                        if train:
                            statistics[column]["data_drift"] = detect_data_drift(
                                data=data,
                                column=column,
                                n_splits=self.config.data_drift.n_splits,
                                dimension=self.config.data_drift.dimension,
                                n_iteration=self.config.data_drift.n_iterations,
                                path=self.report // "images" // name // file,
                                seed=int(os.environ["PYTHONHASHSEED"]),
                            )
                        else:
                            statistics[column]["data_drift"] = detect_data_drift(
                                data=train_data,
                                column=column,
                                n_splits=self.config.data_drift.n_splits,
                                dimension=self.config.data_drift.dimension,
                                n_iteration=self.config.data_drift.n_iterations,
                                path=self.report // "images" // name // file,
                                current_data=data,
                                seed=int(os.environ["PYTHONHASHSEED"]),
                            )
                elif column in schema.filepath:
                    pass
                else:
                    e = ValidationError(message=f"Invalid data in column '{column}'")
                    logger.error(e)

                # if data[column].apply(is_valid_filepath).values.all():
                #     print(f"Column {column} is a valid column for filepath")
            self.status[str(name)][str(file)]["statistics"] = statistics

        except Exception as e:
            self.validation_status = False
            e = ValidationError(message=e)
            logger.error(e)
            raise e

        return data, None

    def generate_report(
        self,
    ):
        pass
