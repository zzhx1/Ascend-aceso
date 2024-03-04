#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=12
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model load ckpt path"



TP=8
PP=12

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 70 \
    --hidden-size 14336 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 112 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --vocab-file ${TOKENIZER_MODEL} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --embed-layernorm \
    --padded-vocab-size 250880 \
    --make-vocab-size-divisible-by 1 \
    --attention-softmax-in-fp32 \
    --apply-query-key-layer-scaling \
    --lr 1.2e-4 \
    --train-iters 200 \
    --init-method-std 0.0048 \
    --hidden-dropout 0.0 \
    --position-embedding-type alibi \
    --normalization LayerNorm \
    --no-masked-softmax-fusion \
    --min-lr 6e-6 \
    --lr-decay-iters 200 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42
"

DATA_ARGS="
    --data-path $DATA_PATH
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR