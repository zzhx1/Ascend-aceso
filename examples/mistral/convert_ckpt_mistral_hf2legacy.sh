# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置你需要的并行参数
python tools/checkpoint/convert_ckpt.py \
   --model-type GPT \
   --loader llama2_hf \
   --saver megatron \
   --load-dir ./model_from_hf/Mistral-hf/ \
   --save-dir ./model_weights/Mistral-legacy/ \
   --tokenizer-model ./model_from_hf/Mistral-hf/tokenizer.model \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1