import os

os.system("pip3 install -r /kaggle/input/jigsaw/requirements.txt -t /kaggle/working/")
os.system(
    "pip3 install --no-deps -t /kaggle/working /kaggle/input/jigsaw/jigsaw-0.1.*-py3-none-any.whl"
)
