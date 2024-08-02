# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置你需要的并行参数
python tools/checkpoint/convert_ckpt.py \
   --model-type GPT \
   --loader hf_mcore \
   --saver mg_mcore \
   --load-dir ./model_from_hf/Mistral-hf/ \
   --save-dir ./model_weights/Mistral-mcore/ \
   --tokenizer-model ./model_from_hf/Mistral-hf/tokenizer.model \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --use-mcore-models \
   --model-type-hf llama2 \