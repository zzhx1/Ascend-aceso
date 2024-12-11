
注释
model_ops_info #38 #修改算子数量
kw_args['flex_recompute_activations'] = args.flex_recompute_activations
self.flex_recompute_activations = flex_config.flex_recompute_activations
args.flex_recompute_activations = config_dict["flex_recompute_activations"]
if self.flex_recompute_activations:

添加：
"flex_recompute_activations": [
    true,
    true,
    true
],

```python
adaptive_hyper_parameters ....................... 5
  add_action_finetune_algo ........................ False
  add_action_finetune_dim ......................... False
  add_action_tp_dp_exchange ....................... False
  add_action_tune_tp_dp ........................... False
  check_recompute_with_group ...................... False
  config_save_path ................................ ./test_eval_logs_4/configs/gpt/1_3B/
  config_suffix ................................... 2024-12-7-15-48-59
  consider_collective_memory ...................... False
  consider_reserved_space ......................... True
  consider_shared_space ........................... True
  continue_when_fail .............................. True
  decoder_seq_len ................................. 512
  end_num_stages .................................. 8
  finetune_after_trial ............................ 0
  finetune_tp_dp_after_trial ...................... False
  flex_recompute .................................. True
  forbid_turn_back ................................ False
  global_batch_size ............................... 1024
  high_memory_rate ................................ 0.9
  init_dim ........................................ tp
  initial_point ................................... balance
  log_path ........................................ ./test_eval_logs_4/search/gpt/1_3B/
  max_num_hops .................................... 7
  max_num_trials .................................. 100
  max_op_move_steps ............................... 5
  max_tp .......................................... 8
  memory_limit .................................... 28000
  memory_main_params .............................. 2
  memory_optimizer ................................ 4
  memory_pred_type ................................ MAX
  micro_batch_size ................................ [1, 2, 4, 8]
  min_mbs ......................................... 1
  model_name ...................................... gpt
  model_size ...................................... 1_3B
  multi_process ................................... True
  num_algos ....................................... 2
  num_gpus ........................................ 8
  num_gpus_per_node ............................... 8
  num_layers ...................................... 24
  num_nodes ....................................... 1
  num_of_saved_configs ............................ 1
  num_partners_in_op_mig .......................... 1
  only_top_1_target ............................... False
  op_group_size ................................... 1
  peak_mem_in_backward ............................ 0
  predict_delta_time .............................. False
  print_debug_info ................................ False
  print_gpu_mig_details ........................... False
  print_move_op_details ........................... False
  print_recomp_debug_info ......................... False
  print_recompute_ops ............................. True
  profiled_time_path .............................. ../profiler/profiled-time-miniset/
  random_order_actions ............................ False
  resharding ...................................... True
  save_to_csv ..................................... None
  seq_len ......................................... 2048
  simple_prim_mbs ................................. False
  simple_prim_mig ................................. False
  sort_metric ..................................... max_stage_time
  start_num_stages ................................ 1
  support_comm_predict ............................ False
  time_budget_per_trial ........................... 200
  time_budget_total ............................... 200
```



```python
print(full_op_list)
'dec-embedding', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process', 
'dec-self-attention', 'dec-mlp', 'dec-post-process']
```







```python
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=0, num_gpus=8, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=1, history='start|129233.49|6396.31| op = [50]| tp = [8, ] | dp = [1, ] | algo = [0, ] | rc = [0] | gpus = [8] | micro_bs = 1 | time = [129233] | memory = [6396]\n', time_list=[129233.4928007565], memory_list=[6396.3110000000015], compute_time_list=[126.204114], total_gpu_time=1032858.3057435461, breakdown_ideal_time_per_gpu=[42.029100500000006], breakdown_eff_loss_time_per_gpu=[84.1750135], breakdown_recomp_time_per_gpu=[0.0], efficient_time_list=[0.0], adaptive_times=0)


AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=1, num_gpus=4, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=0, num_gpus=4, ops=['dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=2, history='start|74822.38|9232.99| op = [25, 25]| tp = [4, 4, ] | dp = [1, 1, ] | algo = [0, 0, ] | rc = [0, 0] | gpus = [4, 4] | micro_bs = 1 | time = [72100, 74822] | memory = [9232, 6244]\n', time_list=[72100.92140351873, 74822.38189657548], memory_list=[9232.992999999999, 6244.503], compute_time_list=[69.42916900000002, 72.08992], total_gpu_time=586546.4947355957, breakdown_ideal_time_per_gpu=[40.95076300000001, 43.107438], breakdown_eff_loss_time_per_gpu=[28.478406000000003, 28.98248200000001], breakdown_recomp_time_per_gpu=[0.0, 0.0], efficient_time_list=[6372.806030262493, 0.0], adaptive_times=0)


AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=2, num_gpus=4, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=1, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=0, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=3, history='start|87216.65|11496.62| op = [16, 16, 18]| tp = [4, 2, 2, ] | dp = [1, 1, 1, ] | algo = [0, 0, 0, ] | rc = [0, 0, 0] | gpus = [4, 2, 2] | micro_bs = 1 | time = [47224, 80235, 87216] | memory = [8248, 11496, 9024]\n', time_list=[47224.83862480315, 80235.5748183377, 87216.65147885989], memory_list=[8248.962999999998, 11496.618999999999, 9024.362], compute_time_list=[45.04795700000001, 76.406864, 84.14111499999999], total_gpu_time=522129.9118505512, breakdown_ideal_time_per_gpu=[26.963436499999997, 53.92398799999999, 60.26554099999999], breakdown_eff_loss_time_per_gpu=[18.084520500000014, 22.48287600000001, 23.875574], breakdown_recomp_time_per_gpu=[0.0, 0.0, 0.0], efficient_time_list=[94643.60428384252, 9786.917259039516, 0.0], adaptive_times=0)


AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=3, num_gpus=2, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=2, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=1, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=0, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=4, history='start|67708.37|14936.97| op = [12, 12, 12, 14]| tp = [2, 2, 2, 2, ] | dp = [1, 1, 1, 1, ] | algo = [0, 0, 0, 0, ] | rc = [0, 0, 0, 0] | gpus = [2, 2, 2, 2] | micro_bs = 1 | time = [59380, 60727, 60727, 67708] | memory = [14936, 11680, 8840, 7312]\n', time_list=[59380.718671087554, 60727.291861622114, 60727.291861622114, 67708.36852214429], memory_list=[14936.971, 11680.698999999999, 8840.458999999999, 7312.282], compute_time_list=[56.898502, 57.305148, 57.305148, 65.039399], total_gpu_time=495151.26649962034, breakdown_ideal_time_per_gpu=[40.44587599999999, 40.44299099999999, 40.44299099999999, 46.784544], breakdown_eff_loss_time_per_gpu=[16.452626000000013, 16.862157000000007, 16.862157000000007, 18.25485500000001], breakdown_recomp_time_per_gpu=[0.0, 0.0, 0.0, 0.0], efficient_time_list=[11783.980860237996, 9765.321919745102, 9765.321919745102, 0.0], adaptive_times=0)
```



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



修改函数：
1. get_tunable_op_list：无需修改，只在finetune阶段使用该函数，搜索算法未启用finetune，

2. [get_no_recompute_op_list](search/aceso_cost_model.py#L13)
ops_not_recomputed = get_no_recompute_op_list(args) -> get_next_recompute_op_group() -> check_recompute()

**check_recompute()**
->[predict_value_after_move](search/aceso_cost_model.py#L738)





--position-embedding-type rope \
加了这个参数之后，会出现 AssertionError: context parallel group is not initialized