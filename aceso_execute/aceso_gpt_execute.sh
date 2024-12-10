#! /bin/bash
export COMBINED_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ROOT_PATH=$(pwd)
model_name=gpt
#### Model info ####
model_size=1_3B
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

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

#### vocab info ####
VOCAB_FILE="./vocab_file/gpt2-vocab.json"
MERGE_FILE="./vocab_file/gpt2-merges.txt"
DATA_ARGS="
    --mock-data \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 949,50,1
"

#### Paths ####
RESULT_PATH=${ROOT_PATH}/aceso_execute/logs/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv



for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) start executing config: $config_name ." >> ${RESULT_PATH}full_log.log



torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --use-cp-send-recv-overlap \
    --context-parallel-algo megatron_cp_algo \
    --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
    --train-iters 10 \
    --eval-iters 0 \
    --mock-data \
    --vocab-file vocab_file/gpt2-vocab.json \
    --merge-file vocab_file/gpt2-merges.txt \
    --split 949,50,1 \
    --distributed-backend nccl \
    --transformer-impl local \
    --no-async-tensor-model-parallel-allreduce \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --tokenizer-type GPT2BPETokenizer \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --fp16 \
    --use-flash-attn \
    --log-interval 1 \
    --log-path $LOG_PATH \
    2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 

# --lr-warmup-fraction .001 \
# --use-flash-attn \
# --attention-softmax-in-fp32 \
# --initial-loss-scale 4096 \
# --no-shared-storage \

    # --no-shared-storage \
    # --no-masked-softmax-fusion \
    # --no-bias-gelu-fusion \