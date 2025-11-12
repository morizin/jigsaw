while [[ '$(kaggle kernels status morizin/jigsaw-wheels | egrep -o "[A-Z]{2,}")' == "RUNNING" ]]; do
    pass
done

pip install --no-index --find-links=wheels/ -r requirements.txt

