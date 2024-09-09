# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Yi-hf/ \
    --save-dir ./modelweights/Yi-mcore/ \
    --tokenizer-model ./model_from_hf/Yi-hf/tokenizer.model
