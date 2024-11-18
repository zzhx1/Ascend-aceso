# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union
import inspect

import torch
from torch.autograd.variable import Variable
import torch.distributed

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type, allocate_mem_buff
from megatron.training.global_vars import get_num_microbatches, get_args

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None and parallel_state.get_virtual_pipeline_model_parallel_world_size() > 1:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            args = get_args()
            assert get_num_microbatches() % pipeline_model_parallel_size == 0, \
                    'number of microbatches is not divisible by pipeline-parallel ' \
                    'size when using interleaved schedule'
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    extra_tensors=None,
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            print(f"forward func: {inspect.getfile(forward_step_func)}:{forward_step_func.__name__}")
            output_tensor, output_extra_tensors, loss_func = forward_step_func(data_iterator, model, extra_tensors)
        else:
            output_tensor, output_extra_tensors, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch, extra_tensors
            )

    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.tensor(1.0, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor, output_extra_tensors
    return [output_tensor, output_extra_tensors]

def update_output_extra_tensors_grad(output_extra_tensors, output_extra_tensors_grad):
    if output_extra_tensors_grad is not None and output_extra_tensors is not None:
        for key in sorted(output_extra_tensors):
            output_extra_tensors[key].grad = output_extra_tensors_grad[key]
    return output_extra_tensors

def retain_input_tensors_grad_and_check_output_grad(input_tensor, extra_tensors, output_tensor, output_tensor_grad):
    if input_tensor is not None:
        for key in sorted(input_tensor):
            input_tensor[key].retain_grad()
    if extra_tensors is not None:
        for key in sorted(extra_tensors):
            if extra_tensors[key].requires_grad:
                extra_tensors[key].retain_grad()
                
    output_tensor_list = None
    output_tensor_grad_list = None
    if output_tensor is not None:
        if isinstance(output_tensor, dict):
            output_tensor_list = []
            for key in sorted(output_tensor):
                if output_tensor[key] is not None:
                    output_tensor_list.append(output_tensor[key])
        else:
            output_tensor_list = output_tensor
    if output_tensor_grad is not None:
        if isinstance(output_tensor_grad, dict):
            output_tensor_grad_list = []
            for key in sorted(output_tensor_grad):
                if output_tensor_grad[key] is not None:
                    output_tensor_grad_list.append(output_tensor_grad[key])
        else:
            output_tensor_grad_list = output_tensor_grad

    return output_tensor_list, output_tensor_grad_list

def collect_grad_of_input_and_extra_tensors(input_tensor, extra_tensors):
    input_tensor_grad = None
    extra_tensors_grad = None
    if input_tensor is not None:
        input_tensor_grad = {}
        for key in sorted(input_tensor):
            if input_tensor[key].grad is None:
                input_tensor_grad[key] = torch.zeros(list(input_tensor[key].size()), requires_grad=False, device=torch.cuda.current_device(), dtype=torch.float16)
            else:
                input_tensor_grad[key] = input_tensor[key].grad

    # When we want to send back the gradients of some extra tensors (encoder_output),
    # its gradients may not be calculated yet, current workaround: send back zero values.
    if extra_tensors is not None:
        extra_tensors_grad = {}
        for key in sorted(extra_tensors):
            if extra_tensors[key].grad is None:
                extra_tensors_grad[key] = torch.zeros(list(extra_tensors[key].size()), requires_grad=False, device=torch.cuda.current_device(), dtype=torch.float16)
            else:
                extra_tensors_grad[key] = extra_tensors[key].grad

    return input_tensor_grad, extra_tensors_grad


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, extra_tensors=None, output_extra_tensors=None, output_extra_tensors_grad=None, model=None):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()
    
    output_extra_tensors = update_output_extra_tensors_grad(output_extra_tensors, output_extra_tensors_grad)
    output_tensor, output_tensor_grad = retain_input_tensors_grad_and_check_output_grad(input_tensor, extra_tensors, output_tensor, output_tensor_grad)

    # Backward pass.
    if output_tensor_grad is None and config.grad_scale_func is not None:       
        output_tensor = config.grad_scale_func(output_tensor)
    
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    extra_tensors_grad = None

    input_tensor_grad, extra_tensors_grad = collect_grad_of_input_and_extra_tensors(input_tensor, extra_tensors)

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad, extra_tensors_grad


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, _ = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            )
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, _ = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
    )

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=model)

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func([model])

    return forward_data_store


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
    
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []

    input_extra_tensors_list = [[] for _ in range(len(model))]
    output_extra_tensors_list = [[] for _ in range(len(model))]

    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
        output_extra_tensor_grads_list = [[] for _ in range(len(model))]

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    if config.context_parallel_size > 1:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def list_append_helper(tensor_list, model_chunk_id, new_tensor, name=None):
        tensor_list[model_chunk_id].append(new_tensor)      
    
    def list_pop_helper(tensor_list, model_chunk_id, name=None):
        popped_tensor = tensor_list[model_chunk_id].pop(0)
        return popped_tensor

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id: int, checkpoint_activations_microbatch: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                list_append_helper(input_tensors, model_chunk_id, None, "input_tensor")
                list_append_helper(input_extra_tensors_list, model_chunk_id, None, "input_extra_tensors")
        input_tensor = input_tensors[model_chunk_id][-1]
        extra_tensors = input_extra_tensors_list[model_chunk_id][-1]

        output_tensor, output_extra_tensors = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
            extra_tensors=extra_tensors,
        )
        list_append_helper(output_tensors, model_chunk_id, output_tensor, "output_tensor")
        list_append_helper(output_extra_tensors_list, model_chunk_id, output_extra_tensors, "output_extra_tensors")

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor, output_extra_tensors

    def backward_step_helper(microbatch_id) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                list_append_helper(output_tensor_grads, model_chunk_id, None, "output_tensor_grad")
                list_append_helper(output_extra_tensor_grads_list, model_chunk_id, None, "output_extra_tensor_grads")
        input_tensor = list_pop_helper(input_tensors, model_chunk_id, "input_tensor")
        output_tensor = list_pop_helper(output_tensors, model_chunk_id, "output_tensor")
        output_tensor_grad = list_pop_helper(output_tensor_grads, model_chunk_id, "output_tensor_grad")
        
        extra_tensors = list_pop_helper(input_extra_tensors_list, model_chunk_id, "input_extra_tensors")
        output_extra_tensors = list_pop_helper(output_extra_tensors_list, model_chunk_id, "output_extra_tensors")
        output_extra_tensor_grads = list_pop_helper(output_extra_tensor_grads_list, model_chunk_id, "output_extra_tensor_grads")
        
        input_tensor_grad, extra_tensors_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config, extra_tensors, output_extra_tensors, output_extra_tensor_grads
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad, extra_tensors_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensor, extra_tensors = p2p_communication.recv_forward(tensor_shape, config)
    list_append_helper(input_tensors, 0, input_tensor, "input_tensor")
    list_append_helper(input_extra_tensors_list, 0, extra_tensors, "input_extra_tensors")

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, output_extra_tensors = forward_step_helper(k, checkpoint_activations_microbatch)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
            output_extra_tensors = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                extra_tensors_grad = None
                output_tensor_grad = None
                output_extra_tensors_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    extra_tensors,
                    output_tensor_grad,
                    output_extra_tensors_grad
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    output_extra_tensors=output_extra_tensors,
                    extra_tensors_grad=extra_tensors_grad,
                )
                list_append_helper(output_tensor_grads, num_model_chunks - 1, output_tensor_grad, "output_tensor_grad")
                list_append_helper(output_extra_tensor_grads_list, num_model_chunks - 1, output_extra_tensors_grad, "output_extra_tensor_grads")
            else:
                input_tensor, extra_tensors = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config, output_extra_tensors=output_extra_tensors
                )
            list_append_helper(input_tensors, next_forward_model_chunk_id, input_tensor, "input_tensor")
            list_append_helper(input_extra_tensors_list, next_forward_model_chunk_id, extra_tensors, "input_extra_tensors")
        else:
            raise NotImplementedError("Overlap p2p communication is not supported with interleaved schedule")
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        if config.overlap_p2p_comm:
            raise NotImplementedError("Overlap p2p communication is not supported with interleaved schedule")
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(forward_k, checkpoint_activations_microbatch)

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor, output_extra_tensors = forward_step_helper(forward_k, checkpoint_activations_microbatch)

            # Backward pass.
            backward_k = k
            input_tensor_grad, extra_tensors_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                extra_tensors,
                output_tensor_grad,
                output_extra_tensors_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                output_extra_tensors=output_extra_tensors,
                extra_tensors_grad=extra_tensors_grad,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            list_append_helper(input_tensors, next_forward_model_chunk_id, input_tensor, "input_tensor")
            list_append_helper(input_extra_tensors_list, next_forward_model_chunk_id, extra_tensors, "input_extra_tensors")
        if recv_next:
            list_append_helper(output_tensor_grads, next_backward_model_chunk_id, output_tensor_grad, "output_tensor_grad")
            list_append_helper(output_extra_tensor_grads_list, next_backward_model_chunk_id, output_extra_tensors_grad, "output_extra_tensor_grads")

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            raise NotImplementedError("Overlap p2p communication is not supported with interleaved schedule")
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grad, output_extra_tensors_grad = p2p_communication.recv_backward(tensor_shape, config=config)
            list_append_helper(output_tensor_grads, num_model_chunks - 1, output_tensor_grad, "output_tensor_grad")
            list_append_helper(output_extra_tensor_grads_list, num_model_chunks - 1, output_extra_tensors_grad, "output_extra_tensor_grads")
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad, extra_tensors_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grad, output_extra_tensors_grad = p2p_communication.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config, extra_tensors_grad=extra_tensors_grad
            )
            if recv_next:
                list_append_helper(output_tensor_grads, next_backward_model_chunk_id, output_tensor_grad, "output_tensor_grad")
                list_append_helper(output_extra_tensor_grads_list, next_backward_model_chunk_id, output_extra_tensors_grad, "output_extra_tensor_grads")
            # [NOT SURE]
            # output_tensor_grads[next_backward_model_chunk_id].append()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(model)

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    if config.context_parallel_size > 1:
        seq_length = seq_length // parallel_state.get_context_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    input_tensors_list, input_extra_tensors_list = [], []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors_list.append(None)
            input_extra_tensors_list.append(None)
        else:
            input_tensors, input_extra_tensors = p2p_communication.recv_forward(tensor_shape, config)
            input_tensors_list.append(input_tensors)
            input_extra_tensors_list.append(input_extra_tensors)
    return input_tensors_list, input_extra_tensors_list


def recv_backward(tensor_shapes, config):
    output_tensor_grads, output_extra_tensors_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
            output_extra_tensors_grads.append(None)
        else:
            output_tensor_grad, output_extra_tensor_grad = p2p_communication.recv_backward(tensor_shape, config)
            output_tensor_grads.append(output_tensor_grad)
            output_extra_tensors_grads.append(output_extra_tensor_grad)
    return output_tensor_grads, output_extra_tensors_grads


def send_forward(output_tensors, tensor_shapes, config, output_extra_tensors_list=None):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
        output_extra_tensors_list = [output_extra_tensors_list]
    for output_tensor, tensor_shape, output_extra_tensors in zip(output_tensors, tensor_shapes, output_extra_tensors_list):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape, config, output_extra_tensors)


def send_backward(input_tensor_grads, tensor_shapes, config, extra_tensors_grads_list=None):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
        extra_tensors_grads_list = [extra_tensors_grads_list]
    for input_tensor_grad, tensor_shape, extra_tensors_grads in zip(input_tensor_grads, tensor_shapes, extra_tensors_grads_list):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape, config, extra_tensors_grads)


def send_forward_recv_backward(output_tensors, tensor_shapes, config, output_extra_tensors_list=None):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
        output_extra_tensors_list = [output_extra_tensors_list]
    output_tensor_grads = []
    output_extra_tensors_grads = []
    for (output_tensor, tensor_shape, output_extra_tensors) in zip(output_tensors, tensor_shapes, output_extra_tensors_list):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            output_extra_tensors_grads.append(None)
            continue
        output_tensor_grad, output_extra_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config, output_extra_tensors
        )
        output_tensor_grads.append(output_tensor_grad)
        output_extra_tensors_grads.append(output_extra_tensor_grad)
    return output_tensor_grads, output_extra_tensors_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config, extra_tensors_grads = None):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
        extra_tensors_grads = [extra_tensors_grads]
    input_tensors = []
    extra_tensors_list = []
    for (input_tensor_grad, tensor_shape, extra_tensors_grad) in zip(input_tensor_grads, tensor_shapes, extra_tensors_grads):
        if tensor_shape is None:
            input_tensors.append(None)
            extra_tensors_list.append(None)
            continue
        input_tensor, extra_tensors = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config, extra_tensors_grad
        )
        input_tensors.append(input_tensor)
        extra_tensors_list.append(extra_tensors)
    return input_tensors, extra_tensors_list


def delete_tensors(tensor_dict):
    if tensor_dict is not None:
        saved_keys = []
        for key in tensor_dict:
            saved_keys.append(key)
        for key in saved_keys:
            del tensor_dict[key]
        del tensor_dict

def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    losses_reduced = None
    
    input_extra_tensors_list = None
    output_extra_tensors_list = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
        input_extra_tensors_list = []
        output_extra_tensors_list = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor, extra_tensors = p2p_communication.recv_forward(recv_tensor_shapes, config)
        output_tensor, output_extra_tensors = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            extra_tensors=extra_tensors,
        )
        p2p_communication.send_forward(output_tensor, config, output_extra_tensors=output_extra_tensors)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            input_extra_tensors_list.append(extra_tensors)
            output_extra_tensors_list.append(output_extra_tensors)
            deallocate_output_tensor(output_tensors[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor, extra_tensors = p2p_communication.recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, output_extra_tensors = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            extra_tensors=extra_tensors,
        )

        if forward_only:
            p2p_communication.send_forward(output_tensor, config, output_extra_tensors=output_extra_tensors)

            if not last_iteration:
                input_tensor, extra_tensors = p2p_communication.recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad, output_extra_tensors_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, recv_tensor_shapes, config, output_extra_tensors=output_extra_tensors
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            input_extra_tensors_list.append(extra_tensors)
            output_extra_tensors_list.append(output_extra_tensors)
            deallocate_output_tensor(output_tensors[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            extra_tensors = input_extra_tensors_list.pop(0)
            output_extra_tensors = output_extra_tensors_list.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad, extra_tensors_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config, extra_tensors, output_extra_tensors, output_extra_tensors_grad, model
            )

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, config, extra_tensors_grad=extra_tensors_grad)
            else:
                input_tensor, extra_tensors = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config, extra_tensors_grad=extra_tensors_grad
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            extra_tensors = input_extra_tensors_list.pop(0)
            output_extra_tensors = output_extra_tensors_list.pop(0)

            output_tensor_grad, output_extra_tensors_grad = p2p_communication.recv_backward(send_tensor_shapes, config)

            input_tensor_grad, extra_tensors_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config, extra_tensors, output_extra_tensors, output_extra_tensors_grad, model
            )
            
            delete_tensors(output_extra_tensors)
            delete_tensors(output_extra_tensors_grad)

            p2p_communication.send_backward(input_tensor_grad, config, extra_tensors_grad=extra_tensors_grad)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])

    return forward_data_store


# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None

def init_checkpointed_activations_memory_buffer():
    """Initializ the memory buffer for the checkpointed activations."""
    args = get_args()

    per_layer = args.micro_batch_size * args.max_position_embeddings * \
                args.hidden_size // args.tensor_model_parallel_size
    assert args.num_layers % args.checkpoint_num_layers == 0, \
        'number of layers is not divisible by checkpoint-num-layers'
    num_checkpointer_layers = args.num_layers // args.checkpoint_num_layers
    numel = per_layer * num_checkpointer_layers
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float

    global _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER
    assert _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is None, \
        'checkpointed activations memory buffer is already allocated.'
    _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = allocate_mem_buff(
        'checkpointed activations', numel, dtype, track_usage=False)


def reset_checkpointed_activations_memory_buffer():
    """Reset the memory used for checkpointing."""
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.reset()