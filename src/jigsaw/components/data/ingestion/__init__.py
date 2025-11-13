from ....core import DataIngestionConfig, DataIngestionArtifact
from ....errors import ComponentError
from typeguard import typechecked
from ....constants import LENGTH
from ... import Component
from .... import logger
import subprocess
import kagglehub
import shutil


class DataIngestionComponent(Component):
    @typechecked
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config: DataIngestionConfig = config
            self.names: list[str] = list(self.config.sources.keys())
            length = max(LENGTH, max(list(map(len, self.names))))
            print("=" * length)
            print("|", "Datasets Available".center(length - 4), "|")
            print("=" * length)
            for name in self.names:
                print("|", name.center(length - 4), "|")
            print("=" * length)

        except Exception as e:
            logger.error(e)
            raise e

    def __call__(self) -> DataIngestionArtifact:
        names: list[str] = []
        for name, datasource in self.config.sources.items():
            target_path = self.config.outdir // name
            if datasource.source.lower().strip() == "local":
                if datasource.type.lower().strip() == "dir":
                    return_code = shutil.copytree(
                        datasource.uri, str(target_path), dirs_exist_ok=True
                    )
                elif datasource.type.lower().strip() == "file":
                    return_code = shutil.copy(datasource.uri, str(target_path))

                if return_code:
                    names.append(name)

            elif datasource.source.lower().strip() == "kaggle":
                if datasource.type.lower().strip() == "competition":
                    path = kagglehub.competition_download(datasource.name)
                    logger.info(
                        f"Downloading {datasource.uri} competition dataset to {str(target_path.path)}"
                    )

                elif datasource.type.lower().strip() == "dataset":
                    path = kagglehub.dataset_download(datasource.uri)
                    logger.info(
                        f"Downloading {datasource.uri} kaggle dataset to {str(target_path.path)}"
                    )

                else:
                    e = ComponentError(
                        self,
                        f"Invalid type '{datasource.type}' for source {datasource.source}",
                    )
                    logger.error(e)
                    raise e

                return_code = subprocess.call(
                    f"mv {path.rstrip('/') + '/*'} {target_path}/",
                    shell=True,
                )
                if return_code == 0:
                    names.append(name)

                subprocess.call(f"rm -rf {path}".split())

            else:
                e = ComponentError(self, f"Invalid source '{datasource.source}'")
                logger.error(e)
                raise e

        return DataIngestionArtifact(path=self.config.outdir, names=names)
