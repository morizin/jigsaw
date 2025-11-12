
if [ ! -f "./working/kernel-metadata.json" ]; then
    kaggle kernels init -p ./working/
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i "" "s/" ./working/kernel-metadata.json