#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="./dataset/gpt_text_sentence"
VOCAB_FILE="./vocab_file/gpt2-vocab.json"
MERGE_FILE="./vocab_file/gpt2-merges.txt"
# CKPT_LOAD_DIR="your model ckpt path"
# CKPT_SAVE_DIR="your save ckpt path"

TP=2
PP=2
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=24
SEQ_LEN=2048
MBS=8
GBS=1024
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=32

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK 
"

# MOE_ARGS="
#     --num-experts 4 \
#     --expert-model-parallel-size ${EP} \
#     --moe-router-topk 2 \
#     --moe-router-load-balancing-type aux_loss \
#     --moe-aux-loss-coeff 0.01 \
#     --moe-permutation-async-comm \
#     --disable-bias-linear \
#     --moe-expert-capacity-factor 1.1 \
#     --moe-token-dispatcher-type alltoall \
#     --moe-pad-expert-input-to-capacity
# "

GPT_ARGS="
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 10 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
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
    --fp16
"

DATA_ARGS="
    --mock-data \
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 949,50,1
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_gpt4_mcore_moe_drop_tp${TP}_pp${PP}_ep${EP}_cp${CP}_layer${NUM_LAYERS}.log
    
