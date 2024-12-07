
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
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=0, num_gpus=8, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=1, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=1, num_gpus=4, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=0, num_gpus=4, ops=['dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=2, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 3
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=2, num_gpus=4, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=1, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=0, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=3, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 4
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=3, num_gpus=2, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=2, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=1, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=0, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=4, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 5
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=4, num_gpus=2, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=3, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=2, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=1, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=4, num_stages_behind=0, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=5, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 6
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=5, num_gpus=2, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=4, num_gpus=2, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=3, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=2, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=4, num_stages_behind=1, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=5, num_stages_behind=0, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], num_stages=6, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 7
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=6, num_gpus=2, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[2, 2, 2, 2, 2, 2, 2], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=5, num_gpus=1, ops=['dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=4, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=3, num_gpus=1, ops=['dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=4, num_stages_behind=2, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=5, num_stages_behind=1, num_gpus=1, ops=['dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=6, num_stages_behind=0, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0])], num_stages=7, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
working on num_stages = 8
AcesoConfig(global_bs=1024, micro_bs=1, stages=[AcesoStageInfo(index=0, num_stages_behind=7, num_gpus=1, ops=['dec-embedding', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=1, num_stages_behind=6, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=2, num_stages_behind=5, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=3, num_stages_behind=4, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=4, num_stages_behind=3, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=5, num_stages_behind=2, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=6, num_stages_behind=1, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention'], recompute_ops=[0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0]), AcesoStageInfo(index=7, num_stages_behind=0, num_gpus=1, ops=['dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-self-attention', 'dec-mlp', 'dec-post-process'], recompute_ops=[0, 0, 0, 0, 0, 0, 0, 0], tp_size=[1, 1, 1, 1, 1, 1, 1, 1], dp_size=[1, 1, 1, 1, 1, 1, 1, 1], algo=[0, 0, 0, 0, 0, 0, 0, 0])], num_stages=8, history='', time_list=[], memory_list=[], compute_time_list=[], total_gpu_time=0, breakdown_ideal_time_per_gpu=[], breakdown_eff_loss_time_per_gpu=[], breakdown_recomp_time_per_gpu=[], efficient_time_list=[], adaptive_times=0)
```