#! /bin/bash
export COMBINED_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ROOT_PATH=$(pwd)
model_name=gpt
#### Model info ####
# model_size=1_3B
model_size=2_6B

#### Hardware info ####
NNODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000

####  config Paths ####
RESULT_PATH=${ROOT_PATH}/aceso_execute/logs/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv

VOCAB_FILE="./vocab_file/gpt2-vocab.json"
MERGE_FILE="./vocab_file/gpt2-merges.txt"

CP=1
CP_TYPE='megatron_cp_algo'

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --use-flash-attn \
    --transformer-impl local \
    --use-fused-rotary-pos-emb \
    --tokenizer-type GPT2BPETokenizer \
    --use-distributed-optimizer \
    --train-iters 10 \
    --eval-iters 0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-shared-storage \
    --fp16 \
    --clip-grad 1.0 \
"


DATA_ARGS="
    --mock-data \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 949,50,1
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
    --log-throughput \
    --log-path $LOG_PATH \
"

for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) start executing config: $config_name ." >> ${RESULT_PATH}full_log.log



torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 
