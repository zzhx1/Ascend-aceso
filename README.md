# [README_origin](./README_origin.md)


## Todo

- [x] aceso runtime

```bash
bash run_aceso.sh
```

- [x] aceso profiler

```bash
# Note: Please modify the Python path in the script
cd profiler
bash script/profiler_small.sh
```

- [x] aceso search algorithm
```bash
cd search
bash script/search_gpt.sh
```

- [] train the model use search result




执行脚本：`bash aceso_execute/aceso_gpt_execute.sh`
aceso搜索出的最佳方案放在了 `aceso_execute/logs/configs/gpt/1_3B/top_configs`



修改:

- 注释掉了`runtime`里面关于 `flex_recompute_activations` 的部分
 - [kw_args['flex_recompute_activations']](megatron/training/arguments.py#L565)
 - [self.flex_recompute_activations](megatron/core/flexmodels/common/flex_model.py#L365)
 - [ args.flex_recompute_activations](megatron/training/json_arguments.py#L36)
 - [if self.flex_recompute_activations:](megatron/core/flexmodels/common/flex_model.py#L610)


- aceso search算法里面的initialize里面get_full_op_list的逻辑。
  - [get_full_op_list](search/model_ops_info.py#L38)，修改索引位置。

目前遇到问题：爆显存， 无法使用flash-attention.
RuntimeError: NPU out of memory. Tried to allocate 2.13 GiB (NPU 0; 60.97 GiB total capacity; 58.81 GiB already allocated; 58.81 GiB current active; 581.60 MiB free; 59.20 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.

尝试修改json文件里面的参数大小，并没有解决爆显存的问题。修改seq_length, hidden_size, global_batch_size会出现矩阵乘维度对不上，修改micro_batch_size不能解决爆显存。

还有尝试sh文件里面的参数，但是也没有解决问题。

