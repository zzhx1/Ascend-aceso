#! /bin/bash

## Settings used in Aceso Paper
## model_size    num_nodes   gpus_per_node  global_batch_size
## 1_3B          1           4              1024
## 2_6B          1           8              1024
## 6_7B          2           8              1024
## 13B           4           8              1024

#### Model info ####
model_name=gpt
model_size=$1
global_batch_size=1024

#### Hardware info ####
num_nodes=1
gpus_per_node=8
memory_limit=65000

#### Search algo parameters ####
budget=200
max_num_hops=7
init_config=balance

#### Paths ####
DATABASE_PATH=../profiler/profiled-time-miniset/
RESULT_PATH=../search_configs/

LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][SEARCH]($CURRENT_TIME) searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
    
python3 aceso_search.py \
    --model-name $model_name \
    --model-size $model_size \
    --global-batch-size $global_batch_size \
    --micro-batch-size 1 2 4 8 16 32 64 \
    --num-nodes $num_nodes \
    --num-gpus-per-node $gpus_per_node \
    --memory-limit $memory_limit \
    --log-path $LOG_PATH \
    --profiled-time-path $DATABASE_PATH \
    --config-save-path $CONFIG_SAVE_PATH \
    --config-suffix $CURRENT_TIME \
    --max-num-hops $max_num_hops \
    --time-budget-total $budget \
    --initial-point $init_config \
    2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log
 
