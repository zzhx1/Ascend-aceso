# Qwen1.5 $\color{black}{\bf\tiny{【社区贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [Qwen1.5-7B](#qwen15-7b)
	- [训练](#训练)
	- [脚本](#脚本)
	- [性能](#性能)
		- [吞吐](#吞吐)
	- [推理](#推理)
	- [评估](#评估)

# Qwen1.5-7B

## 训练

Qwen1.5-7B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

   ```shell
   git clone https://gitee.com/ascend/ModelLink.git
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout -f bcce6f
   cp -r megatron ../ModelLink/
   cd ..
   cd ModelLink
   mkdir logs
   mkdir model_from_hf
   mkdir dataset
   mkdir ckpt
   ```
2. 搭建环境

   ```bash
   # python3.8
   conda create -n test python=3.8
   conda activate test

   # 安装 torch 和 torch_npu
   pip install torch-2.2.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.2.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 安装加速库
   git clone https://gitee.com/ascend/AscendSpeed.git
   cd AscendSpeed
   git checkout 224ae35e8fc96778f957029d1371ddb623452a50
   pip install -r requirements.txt
   pip install -e .
   cd ..

   # 安装其余依赖库
   pip install -r requirements.txt
   ```
   **注意**：transformer版本要4.37.0以上
3. 下载 Qwen1.5-7B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-7B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen1.5-7B/
   cd ./model_from_hf/Qwen1.5-7B/
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00001-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00002-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00003-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00004-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/special_tokens_map.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/vocab.json
   cd ../../
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```shell
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 8 \
       --target-pipeline-parallel-size 1 \
       --make-vocab-size-divisible-by 16 \
       --load-dir ./model_from_hf/Qwen1.5-7B/ \
       --save-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/ \
       --tokenizer-model ./model_from_hf/Qwen1.5-7B/tokenizer.json \
       --add-qkv-bias
   ```

   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

   ```bash
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader megatron \
       --saver megatron \
       --save-model-type save_huggingface_qwen \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1 \
       --save-dir ./model_from_hf/Qwen1.5-7B 		# 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen-7B/mg2hg/
   ```
   
5. 预训练

   5.1 准备数据集

   下载Qwen1.5-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # 下载数据
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..
   
   # 处理数据   
   mkdir ./dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B \
       --output-prefix ./dataset/Qwen1.5-7B/alpaca \
       --tokenizer-type PretrainedFromHF \
       --seq-length 8192 \
       --workers 4 \
       --log-interval 1000
   ```
   
   5.2 预训练

   配置Qwen1.5-7B 预训练脚本: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
   # 设置 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 

   # 根据实际情况配置词表、数据集、模型参数保存路径
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B"
   TOKENIZER_MODEL="./model_from_hf/Qwen1.5-7B"  #词表路径
   DATA_PATH="./dataset/Qwen1.5-7B/alpaca_text_document"  #数据集路径
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
   ```
   多机运行增加参数 `--overlap-grad-reduce`。

   启动 Qwen1.5-7B 预训练脚本: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
    bash examples/qwen15/pretrain_qwen15_7b_ptd.sh
   ```
   **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加 `--no-shared-storage` 参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
6. 微调

   6.1 准备微调数据集

   下载微调数据集 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # 下载数据集
   mkdir finetune_dataset
   cd ./finetune_dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..
   
   # 处理微调数据集  
   mkdir ./finetune_dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B/ \
       --output-prefix ./finetune_dataset/Qwen1.5-7B/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF \
       --handler-name GeneralInstructionHandler \
       --append-eod
   ```
   6.2 全参微调 全参微调的配置脚本基本和预训练脚本一致。

   *区别是数据集，以及增加训练参数`--is-instruction-dataset`，增加微调参数`--finetune`，增加预训练权重加载参数`--load`
   ，使微调从第一步开始，修改tokenizer参数，去掉`--tokenizer-type Llama2Tokenizer`
   和`--tokenizer-model ${TOKENIZER_MODEL}`。*

   修改如下：

   ```bash
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B/"
   DATA_PATH="./finetune_dataset/Qwen1.5-7B/alpaca"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"

   --load ${CKPT_LOAD_DIR} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-type PretrainedFromHF \
   --tokenizer-name-or-path ${TOKENIZER_PATH} \
   --tokenizer-not-use-fast \
   ```
   
   启动微调脚本: examples/qwen15/tune_qwen15_7b_ptd.sh

   ```shell
    bash examples/qwen15/tune_qwen15_7b_ptd.sh
   ```

### 性能

#### 吞吐

Qwen1.5-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

|       设备       |          模型           |  tokens吞吐 (tokens/s/p)   |
|:--------------:|:---------------------:|:------------------------:|
|      NPUs      |      Qwen1.5-7B       |           2862           |
|       参考       |      Qwen1.5-7B       |           2621           |

## 推理

配置 Qwen1.5-7B 推理脚本：examples/qwen15/generate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B"
```

启动Qwen1.5-7B推理脚本

```bash
bash examples/qwen15/generate_qwen15_7b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen1.5_7b_inference.png)

## 评估

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)
和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置Qwen1.5-7B评估脚本: examples/qwen15/evaluate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"  #词表路径
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"  #模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_7b_ptd.sh
```

|  数据集  | 总学科数 | 总问题数  |                          参考准确率                          | NPU准确率 |
|:-----:|:----:|:-----:|:-------------------------------------------------------:|:------:|
| MMLU  |  57  | 14042 |    [61.0](https://qwenlm.github.io/zh/blog/qwen1.5)     |  60.3  |