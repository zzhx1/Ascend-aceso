#! /bin/bash
ROOT_PATH=$(pwd)/
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}../../
bash scripts/profile_small_p2p.sh
bash scripts/profile_small_gpt.sh
# bash scripts/profile_small_t5.sh
# bash scripts/profile_small_resnet.sh