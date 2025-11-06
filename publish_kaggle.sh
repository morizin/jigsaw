#!/bin/bash

echo "The PID of publishing into kaggle is $$"
rm -rf "./dist/jigsaw*"
uv version --bump patch 
version=$(uv version --short)
uv build

if [ $? -eq 0 ]; then
    read -p "Enter Version ($version) Message : " message
    kaggle datasets version -m "Version $version : $message" -p ./dist

    if [ $? -eq 0 ]; then
        echo "The command was successful"
    fi
fi
