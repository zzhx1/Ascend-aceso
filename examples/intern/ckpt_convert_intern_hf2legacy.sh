source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Internlm-hf/ \
    --save-dir ./model_weights/Internlm-legacy/ \
    --tokenizer-model ./model_from_hf/Internlm-hf/tokenizer.model \
    --add-qkv-bias \
    --add-dense-bias