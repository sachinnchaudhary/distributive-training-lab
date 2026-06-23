from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from meshtrain.parallelism.pipeline_parallel.p2p import (
    recv_backward,
    recv_forward,
    recv_virtual_backward,
    recv_virtual_forward,
    send_backward,
    send_forward,
    send_virtual_backward,
    send_virtual_forward,
)

from meshtrain.parallelism.pipeline_parallel.stage import PipelineStage


@dataclass
class PipelineMicrobatchState:
    input_activation: torch.Tensor
    output_activation: torch.Tensor


def _stage_group_is_active(stage: PipelineStage) -> bool:
    return (
        stage.groups.stage_group is not None
        and len(stage.groups.stage_ranks) > 1
    )


def _stage_barrier(stage: PipelineStage) -> None:
    if _stage_group_is_active(stage):
        dist.barrier(group=stage.groups.stage_group)


def pipeline_forward(
    stage: PipelineStage,
    input_tensor: torch.Tensor | None,
    *,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if stage.is_first:
        if input_tensor is None:
            raise ValueError("first pipeline stage requires input_tensor")
        x = input_tensor
    else:
        x = recv_forward(
            activation_shape,
            stage.groups,
            device=device,
            dtype=dtype,
        )

    x = stage.forward_local(x)

    if stage.is_last:
        return x

    send_forward(x, stage.groups)
    return None


def pipeline_forward_backward(
    stage: PipelineStage,
    input_tensor: torch.Tensor | None,
    *,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor | None:
    if stage.is_first:
        if input_tensor is None:
            raise ValueError("first pipeline stage requires input_tensor")
        x = input_tensor
    else:
        x = recv_forward(
            activation_shape,
            stage.groups,
            device=device,
            dtype=dtype,
        )
        x.requires_grad_(True)

    y = stage.forward_local(x)

    if stage.is_last:
        if loss_fn is None:
            raise ValueError("last pipeline stage requires loss_fn")

        loss = loss_fn(y)
        loss.backward()

        if not stage.is_first:
            if x.grad is None:
                raise RuntimeError("missing activation gradient")
            send_backward(x.grad, stage.groups)

        return loss.detach()

    send_forward(y, stage.groups)

    y_grad = recv_backward(
        tuple(y.shape),
        stage.groups,
        device=device,
        dtype=y.dtype,
    )
    y.backward(y_grad)

    if not stage.is_first:
        if x.grad is None:
            raise RuntimeError("missing activation gradient")
        send_backward(x.grad, stage.groups)

    return None


def _validate_microbatch_schedule_inputs(
    stage: PipelineStage,
    input_microbatches: list[torch.Tensor] | None,
    *,
    num_microbatches: int,
    loss_fn: Callable[[torch.Tensor, int], torch.Tensor] | None,
) -> None:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")

    if stage.is_first:
        if input_microbatches is None:
            raise ValueError("first pipeline stage requires input_microbatches")
        if len(input_microbatches) != num_microbatches:
            raise ValueError("input_microbatches length must match num_microbatches")
    elif input_microbatches is not None:
        raise ValueError("non-first pipeline stage should not receive input_microbatches")

    if stage.is_last and loss_fn is None:
        raise ValueError("last pipeline stage requires loss_fn")


def gpipe_forward_backward(
    stage: PipelineStage,
    input_microbatches: list[torch.Tensor] | None,
    *,
    num_microbatches: int,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    _validate_microbatch_schedule_inputs(
        stage,
        input_microbatches,
        num_microbatches=num_microbatches,
        loss_fn=loss_fn,
    )

    states: list[PipelineMicrobatchState] = []

    for microbatch_id in range(num_microbatches):
        if stage.is_first:
            assert input_microbatches is not None
            x = input_microbatches[microbatch_id]
        else:
            x = recv_forward(
                activation_shape,
                stage.groups,
                device=device,
                dtype=dtype,
            )
            x.requires_grad_(True)

        y = stage.forward_local(x)
        states.append(PipelineMicrobatchState(input_activation=x, output_activation=y))

        if not stage.is_last:
            send_forward(y, stage.groups)

    losses: list[torch.Tensor] = []
    for microbatch_id in reversed(range(num_microbatches)):
        _stage_barrier(stage)
        state = states[microbatch_id]
        x = state.input_activation
        y = state.output_activation

        if stage.is_last:
            assert loss_fn is not None
            loss = loss_fn(y, microbatch_id)
            loss.backward()
            losses.append(loss.detach())

            if not stage.is_first:
                if x.grad is None:
                    raise RuntimeError("missing activation gradient")
                send_backward(x.grad, stage.groups)
        else:
            y_grad = recv_backward(
                tuple(y.shape),
                stage.groups,
                device=device,
                dtype=y.dtype,
            )
            y.backward(y_grad)

            if not stage.is_first:
                if x.grad is None:
                    raise RuntimeError("missing activation gradient")
                send_backward(x.grad, stage.groups)

    if stage.is_last:
        losses.reverse()
        return losses

    return None


def one_forward_one_backward(
    stage: PipelineStage,
    input_microbatches: list[torch.Tensor] | None,
    *,
    num_microbatches: int,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    _validate_microbatch_schedule_inputs(
        stage,
        input_microbatches,
        num_microbatches=num_microbatches,
        loss_fn=loss_fn,
    )

    states: dict[int, PipelineMicrobatchState] = {}
    losses: dict[int, torch.Tensor] = {}

    def forward_one(microbatch_id: int) -> None:
        if stage.is_first:
            assert input_microbatches is not None
            x = input_microbatches[microbatch_id]
        else:
            x = recv_forward(
                activation_shape,
                stage.groups,
                device=device,
                dtype=dtype,
            )
            x.requires_grad_(True)

        y = stage.forward_local(x)
        states[microbatch_id] = PipelineMicrobatchState(
            input_activation=x,
            output_activation=y,
        )

        if not stage.is_last:
            send_forward(y, stage.groups)

    def backward_one(microbatch_id: int) -> None:
        state = states.pop(microbatch_id)
        x = state.input_activation
        y = state.output_activation

        if stage.is_last:
            assert loss_fn is not None
            loss = loss_fn(y, microbatch_id)
            loss.backward()
            losses[microbatch_id] = loss.detach()

            if not stage.is_first:
                if x.grad is None:
                    raise RuntimeError("missing activation gradient")
                send_backward(x.grad, stage.groups)

            return

        y_grad = recv_backward(
            tuple(y.shape),
            stage.groups,
            device=device,
            dtype=y.dtype,
        )
        y.backward(y_grad)

        if not stage.is_first:
            if x.grad is None:
                raise RuntimeError("missing activation gradient")
            send_backward(x.grad, stage.groups)

    num_warmup = min(
        stage.num_stages - stage.stage_index - 1,
        num_microbatches,
    )

    next_forward_id = 0
    for _ in range(num_warmup):
        forward_one(next_forward_id)
        next_forward_id += 1

    next_backward_id = 0
    num_steady_steps = num_microbatches - num_warmup

    for _ in range(num_steady_steps):
        forward_id = next_forward_id
        backward_id = next_backward_id

        if stage.stage_index < stage.num_stages // 2:
            if next_forward_id < num_microbatches:
                forward_one(next_forward_id)
                next_forward_id += 1

            backward_one(backward_id)
            next_backward_id += 1
        else:
            if backward_id not in states:
                if forward_id >= num_microbatches:
                    raise RuntimeError("no forward microbatch available before backward")
                forward_one(forward_id)
                next_forward_id += 1

            backward_one(backward_id)
            next_backward_id += 1

            if next_forward_id < num_microbatches:
                forward_one(next_forward_id)
                next_forward_id += 1

    while next_backward_id < num_microbatches:
        backward_one(next_backward_id)
        next_backward_id += 1

    if stage.is_last:
        return [losses[i] for i in range(num_microbatches)]

    return None


def _validate_virtual_stages(stages: list[PipelineStage]) -> None:
    if not stages:
        raise ValueError("stages must contain at least one PipelineStage")

    for stage in stages:
        if (
            stage.global_virtual_stage_index is None
            or stage.num_virtual_stages is None
            or stage.virtual_stage_index is None
            or stage.virtual_stages_per_rank is None
        ):
            raise ValueError("all stages must be virtual pipeline stages")  


def _virtual_stage_by_global_index(  
        stages: list[PipelineStage], 
) -> dict[int, PipelineStage]:  
      
    _validate_virtual_stages(stages)

    return {
        stage.global_virtual_stage_index: stage
        for stage in stages
    }  


def interleaved_pipeline_forward(
    stages: list[PipelineStage],  
    input_tensor: torch.Tensor | None,  
    *,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:  

    stage_by_index = _virtual_stage_by_global_index(stages)

    first_local_stage = stages[0]
    assert first_local_stage.num_virtual_stages is not None
    num_virtual_stages = first_local_stage.num_virtual_stages

    output: torch.Tensor | None = None  

    for virtual_stage_index in range(num_virtual_stages):  
        stage = stage_by_index.get(virtual_stage_index)  

        if stage is None:  
            continue  
         
        if virtual_stage_index == 0:
            if input_tensor is None:
                raise ValueError(
                    "owner of first virtual pipeline stage requires input_tensor"
                )
            x = input_tensor
        else:
            x = recv_virtual_forward(
                activation_shape,
                stage.groups,
                global_virtual_stage_index=virtual_stage_index,
                device=device,
                dtype=dtype,
            )

        y = stage.forward_local(x)  

        if virtual_stage_index == num_virtual_stages - 1:  
            output = y  
        else:  
           send_virtual_forward(
                y,
                stage.groups,
                global_virtual_stage_index=virtual_stage_index,
                num_virtual_stages=num_virtual_stages,
            )

    return output        


def interleaved_one_forward_one_backward(
    stages: list[PipelineStage],
    input_microbatches: list[torch.Tensor] | None,
    *,
    num_microbatches: int,
    activation_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    _validate_virtual_stages(stages)

    first_local_stage = stages[0]
    assert first_local_stage.num_virtual_stages is not None
    num_virtual_stages = first_local_stage.num_virtual_stages

    first_virtual_owner = 0 in _virtual_stage_by_global_index(stages)
    last_virtual_owner = (num_virtual_stages - 1) in _virtual_stage_by_global_index(stages)

    if num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")
    if first_virtual_owner:
        if input_microbatches is None:
            raise ValueError("owner of first virtual stage requires input_microbatches")
        if len(input_microbatches) != num_microbatches:
            raise ValueError("input_microbatches length must match num_microbatches")
    elif input_microbatches is not None:
        raise ValueError("non-owner of first virtual stage should not receive input_microbatches")
    if last_virtual_owner and loss_fn is None:
        raise ValueError("owner of last virtual stage requires loss_fn")

    stage_by_index = _virtual_stage_by_global_index(stages)
    states: dict[tuple[int, int], PipelineMicrobatchState] = {}
    losses: dict[int, torch.Tensor] = {}

    def forward_virtual_stage(
        global_virtual_stage_index: int,
        microbatch_id: int,
    ) -> None:
        stage = stage_by_index.get(global_virtual_stage_index)
        if stage is None:
            return

        if global_virtual_stage_index == 0:
            assert input_microbatches is not None
            x = input_microbatches[microbatch_id]
        else:
            x = recv_virtual_forward(
                activation_shape,
                stage.groups,
                global_virtual_stage_index=global_virtual_stage_index,
                device=device,
                dtype=dtype,
            )
            x.requires_grad_(True)

        y = stage.forward_local(x)
        states[(global_virtual_stage_index, microbatch_id)] = PipelineMicrobatchState(
            input_activation=x,
            output_activation=y,
        )

        send_virtual_forward(
            y,
            stage.groups,
            global_virtual_stage_index=global_virtual_stage_index,
            num_virtual_stages=num_virtual_stages,
        )

    def backward_virtual_stage(
        global_virtual_stage_index: int,
        microbatch_id: int,
    ) -> None:
        stage = stage_by_index.get(global_virtual_stage_index)
        if stage is None:
            return

        state = states.pop((global_virtual_stage_index, microbatch_id))
        x = state.input_activation
        y = state.output_activation

        if global_virtual_stage_index == num_virtual_stages - 1:
            assert loss_fn is not None
            loss = loss_fn(y, microbatch_id)
            loss.backward()
            losses[microbatch_id] = loss.detach()
        else:
            y_grad = recv_virtual_backward(
                tuple(y.shape),
                stage.groups,
                global_virtual_stage_index=global_virtual_stage_index,
                num_virtual_stages=num_virtual_stages,
                device=device,
                dtype=y.dtype,
            )
            y.backward(y_grad)

        if global_virtual_stage_index > 0:
            if x.grad is None:
                raise RuntimeError("missing activation gradient")
            send_virtual_backward(
                x.grad,
                stage.groups,
                global_virtual_stage_index=global_virtual_stage_index,
            )

    num_warmup = min(num_virtual_stages - 1, num_microbatches)

    next_forward_id = 0
    for _ in range(num_warmup):
        for virtual_stage_index in range(num_virtual_stages):
            forward_virtual_stage(virtual_stage_index, next_forward_id)
        next_forward_id += 1

    next_backward_id = 0
    num_steady_steps = num_microbatches - num_warmup

    for _ in range(num_steady_steps):
        if next_forward_id < num_microbatches:
            for virtual_stage_index in range(num_virtual_stages):
                forward_virtual_stage(virtual_stage_index, next_forward_id)
            next_forward_id += 1

        for virtual_stage_index in reversed(range(num_virtual_stages)):
            backward_virtual_stage(virtual_stage_index, next_backward_id)
        next_backward_id += 1

    while next_backward_id < num_microbatches:
        for virtual_stage_index in reversed(range(num_virtual_stages)):
            backward_virtual_stage(virtual_stage_index, next_backward_id)
        next_backward_id += 1

    if last_virtual_owner:
        return [losses[i] for i in range(num_microbatches)]

    return None
