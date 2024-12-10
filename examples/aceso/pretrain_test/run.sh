#!/bin/bash
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16


DATA_PATH="./dataset/gpt_text_sentence"
VOCAB_FILE="./vocab_file/gpt2-vocab.json"
MERGE_FILE="./vocab_file/gpt2-merges.txt"

TP=1
PP=1
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=10
SEQ_LEN=1024
MBS=1
GBS=4



DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK
"

DATA_ARGS="
    --mock-data \
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 949,50,1
"

# Model related configuration here, pls not overlap with json config
GPT_ARGS="
    --use-mcore-models \
    --context-parallel-algo  ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --use-distributed-optimizer \
    --train-iters 10 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-shared-storage \
    --bf16
"

FLEX_ARGS="
    --flexpipe-config /workspace/RC4/ModelLink/examples/aceso/pretrain_test/test_pretrain.json \
    --log-path  /workspace/RC4/ModelLink/examples/aceso/aceso/pretrain_test/logs \
"
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
    --log-throughput
"

mkdir -p logs
mkdir -p logs/csv

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $FLEX_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_gpt4_mcore_moe_drop_tp${TP}_pp${PP}_ep${EP}_cp${CP}_layer${NUM_LAYERS}.log
    
