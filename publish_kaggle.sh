#!/bin/bash

echo "The PID of publishing into kaggle is $$"
rm -rf ./dist/jigsaw-0.*
uv version --bump patch 
version=$(uv version --short)
uv build
if [ ! -f "./dist/dataset-metadata.json" ]; then
    kaggle datasets init -p ./dist/
    sed -i 's/INSERT_[A-Z_]*/jigsaw/g' ./dist/dataset-metadata.json
fi

if [ $? -eq 0 ]; then
    read -p "Enter Version ($version) Message : " message
    echo "Version $version : $message"
    kaggle datasets version -m "Version $version : $message" -p ./dist

    if [ $? -eq 0 ]; then
        echo "The command was successful"
    else
        exit 1
    fi
    exit 0
fi
exit 1
