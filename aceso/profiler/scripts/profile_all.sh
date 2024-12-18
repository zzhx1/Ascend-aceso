#!/bin/bash

ROOT_PATH=$(pwd)/

modelsize=("1_3B" "2_6B" "6_7B" "13B")

for model_size in "${modelsize[@]}"
do
  bash ${ROOT_PATH}scripts/profile_small_gpt.sh "$model_size"
done