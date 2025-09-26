from pathlib import Path
import subprocess 
import os
import logging

logging.basicConfig(level = logging.INFO, format = '[%(asctime)s: %(message)s')

subprocess.call(['uv', 'init'])
project_name : str = "jigsaw"
list_of_files : list[str] = [
        f'src/{project_name}/__init__.py',
        f'src/{project_name}/pipelines/__init__.py',
        f'src/{project_name}/utils/__init__.py',
        f'src/{project_name}/utils/common.py',
        f'src/{project_name}/config/__init__.py',
        f'src/{project_name}/config/config.py',
        f'src/{project_name}/components/__init__.py',
        f'src/{project_name}/components/dataset/__init__.py',
        f'src/{project_name}/components/models/__init__.py',
        f'src/{project_name}/components/data/__init__.py',
        f'src/{project_name}/components/engines/__init__.py',
        f'src/{project_name}/components/loss/__init__.py',
        f'src/{project_name}/constants/__init__.py',
        f'src/{project_name}/entity/__init__.py',
        f'src/{project_name}/entity/common.py',
        "Dockerfile",
        "docker-compose.yaml",
        ".github/workflow/.gitkeep",
        ".gitignore",
        "research/research.ipynb", 
        "params.yaml",
        "schema.yaml",
        "main.py",
        "config/config.yaml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file : {filename}")
    if not filepath.exists() or os.path.getsize(filepath) == 0:
        subprocess.call(['touch', filepath.as_posix()])
        logging.info(f"Creating empty file : {filepath}")

    else:
        logging.info(f"{filepath} already exists")

subprocess.call("rm -f hello.py".split())

