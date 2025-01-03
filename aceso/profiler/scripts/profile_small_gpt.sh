#! /bin/bash
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ROOT_PATH=$(pwd)/
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}../../

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-miniset/
VOCAB_FILE=${RUNTIME_PATH}../../vocab_file/gpt2-vocab.json
MERGE_FILE=${RUNTIME_PATH}../../vocab_file/gpt2-merges.txt


MAX_NUM_GPUS=8
MODEL_NAME=gpt
MODEL_SIZE=$1
echo "profiling model size: $MODEL_SIZE"
#  num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype are fake.
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=16
CP=1
CP_TYPE='megatron_cp_algo'

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
    --split 949,50,1
"

GPT_ARGS="
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
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

mkdir -p ${PROFILING_PATH}

echo "------ Start profiling ------"
for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
do
    GPUS_PER_NODE=${tp_size}
    DISTRIBUTED_ARGS="
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
    "

    echo [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
    torchrun $DISTRIBUTED_ARGS op_profiler.py \
        ${DATA_ARGS} \
        ${GPT_ARGS} \
        --use-mcore-models \
        --prof-op \
        --prof-tp-size $tp_size \
        --prof-path $PROFILING_PATH \
        --prof-model-name $MODEL_NAME \
        --prof-model-size $MODEL_SIZE \
        --prof-warmup-times 5 \
        --prof-repeat-times 5 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log
    
    echo [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
done


echo "comm_profiler.py"
for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
do
echo [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

python3 comm_profiler.py \
    --prof-path $PROFILING_PATH \
    --prof-cache-file ${PROFILING_PATH}comm_profile.pkl \
    --prof-op-time-path $PROFILING_PATH \
    --prof-tp-size $num_gpus \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-warmup-times 5 \
    --prof-repeat-times 20 \
    --max-data-size 4096 \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

echo [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

done

