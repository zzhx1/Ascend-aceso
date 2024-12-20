#!/bin/bash

ROOT_PATH=$(pwd)/

modelsize=("2_6B" "6_7B" "13B")

for model_size in "${modelsize[@]}"
do
  bash ${ROOT_PATH}scripts/search_gpt.sh "$model_size"
done
