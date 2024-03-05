# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
"""Transformer."""
import torch
import torch.nn.functional as F

from ascendspeed.components.moe.moe import MoE
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_expert_parallel_group

from .module import MegatronModule


class MixtralParallelMLPBM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.ffn_hidden_size
        self.hidden_dim = config.hidden_size

        self.w1 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.w2 = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
        )

        self.w3 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0]
        current_hidden_states = self.w2(current_hidden_states)[0]
        return current_hidden_states


class SwitchMLP(MegatronModule):
    def __init__(self, config, layer_number=None):
        from megatron.model.transformer import ParallelMLP

        super().__init__()
        args = get_args()

        try:
            expert_parallel_group = get_expert_parallel_group()
        except AttributeError:
            expert_parallel_group = None

        if layer_number is None:
            self.block = MoE(
                args.hidden_size,
                MixtralParallelMLPBM(config, ),
                num_experts=args.num_experts,
                ep_size=args.expert_model_parallel_size,
                k=args.moe_router_topk,
                capacity_factor=args.moe_train_capacity_factor,
                eval_capacity_factor=args.moe_train_capacity_factor,
                aux_loss_coef=args.moe_aux_loss_coeff,
                ep_group=expert_parallel_group,
                noisy_gate_policy=args.noisy_gate_policy
            )
        else:
            if layer_number % args.expert_interval == 0:
                self.block = MoE(
                    args.hidden_size,
                    MixtralParallelMLPBM(config, ),
                    num_experts=args.num_experts,
                    ep_size=args.expert_model_parallel_size,
                    k=args.moe_router_topk,
                    capacity_factor=args.moe_train_capacity_factor,
                    eval_capacity_factor=args.moe_train_capacity_factor,
                    aux_loss_coef=args.moe_aux_loss_coeff,
                    ep_group=expert_parallel_group,
                    noisy_gate_policy=args.noisy_gate_policy
                )
            else:
                self.block = ParallelMLP(config)

    def forward(self, hidden_states, used_token=None):
        output = self.block(hidden_states, used_token)
        return output[0], None