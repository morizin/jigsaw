#!/bin/bash

set -ex

echo "The PID of publishing into kaggle is $$"
rm -rf ./dist/jigsaw-0.*
uv version --bump patch 
version=$(uv version --short)
uv build

if [ ! -f "./dist/dataset-metadata.json" ]; then
    kaggle datasets init -p ./dist/
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i "" 's/INSERT_[A-Z_]*/jigsaw/g' ./dist/dataset-metadata.json
    else
        sed -i 's/INSERT_[A-Z_]*/jigsaw/g' ./dist/dataset-metadata.json
    fi
fi

read -p "Enter Version ($version) Message : " message
kaggle datasets version -m "Version $version : $message" -p ./dist

if [ ! -f "./working/kernel-metadata.json" ]; then
    kaggle kernels init -p ./working/
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i "" "s/" ./working/kernel-metadata.json