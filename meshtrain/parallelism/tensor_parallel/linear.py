from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from meshtrain.core.distributed.collectives import all_gather_tensor_parallel
from meshtrain.core.distributed.groups import ParallelGroups


def _tp_size(groups: ParallelGroups) -> int:
    return len(groups.tp_ranks)


def _tp_rank(groups: ParallelGroups) -> int:
    return groups.tp_ranks.index(groups.rank)


def _require_divisible(value: int, parts: int, name: str) -> None:
    if parts < 1:
        raise ValueError(f"parts must be at least 1, got {parts}")
    if value % parts != 0:
        raise ValueError(f"{name}={value} must be divisible by tp_size={parts}")


def _slice_last_dim(tensor: torch.Tensor, groups: ParallelGroups) -> torch.Tensor:
    tp_size = _tp_size(groups)
    tp_rank = _tp_rank(groups)
    dim_size = tensor.shape[-1]
    _require_divisible(dim_size, tp_size, "input last dimension")

    local_dim = dim_size // tp_size
    start = tp_rank * local_dim
    end = start + local_dim
    return tensor[..., start:end]


def _debug_tp(groups: ParallelGroups, message: str) -> None:
    if os.environ.get("MESHTRAIN_TP_DEBUG", "0") != "1":
        return
    print(
        f"rank={groups.rank} tp_group={groups.tp_ranks} tp:{message}",
        flush=True,
    )


class _TensorParallelAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, groups: ParallelGroups) -> torch.Tensor:
        ctx.groups = groups
        output = tensor.clone()
        if groups.tp_group is not None and len(groups.tp_ranks) > 1:
            _debug_tp(groups, f"all_reduce_forward_start shape={tuple(output.shape)}")
            work = dist.all_reduce(
                output,
                op=dist.ReduceOp.SUM,
                group=groups.tp_group,
                async_op=True,
            )
            _debug_tp(groups, "all_reduce_forward_launched")
            work.wait()
            _debug_tp(groups, "all_reduce_forward_wait_done")
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def _all_reduce_tensor_parallel_autograd(
    tensor: torch.Tensor,
    groups: ParallelGroups,
) -> torch.Tensor:
    return _TensorParallelAllReduce.apply(tensor, groups)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer sharded along output features.

    PyTorch linear weight shape is [out_features, in_features], so output-feature
    sharding means each TP rank owns a slice of weight dim 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: ParallelGroups,
        *,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.gather_output = gather_output

        tp_size = _tp_size(groups)
        _require_divisible(out_features, tp_size, "out_features")
        self.local_out_features = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = (
            nn.Parameter(torch.empty(self.local_out_features))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _debug_tp(
            self.groups,
            f"column_linear_start in={self.in_features} out={self.out_features} shape={tuple(x.shape)}",
        )
        local_output = F.linear(x, self.weight, self.bias)
        _debug_tp(self.groups, f"column_linear_done shape={tuple(local_output.shape)}")

        if self.gather_output:
            return all_gather_tensor_parallel(
                local_output,
                self.groups,
                dim=-1,
            )

        return local_output

    @torch.no_grad()
    def load_from_linear(self, linear: nn.Linear) -> None:
        if linear.weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"linear weight shape {tuple(linear.weight.shape)} does not match "
                f"expected {(self.out_features, self.in_features)}"
            )

        tp_rank = _tp_rank(self.groups)
        start = tp_rank * self.local_out_features
        end = start + self.local_out_features

        self.weight.copy_(linear.weight[start:end, :])

        if self.bias is not None:
            if linear.bias is None:
                raise ValueError("source linear has no bias")
            self.bias.copy_(linear.bias[start:end])


class RowParallelLinear(nn.Module):
    """
    Linear layer sharded along input features.

    PyTorch linear weight shape is [out_features, in_features], so input-feature
    sharding means each TP rank owns a slice of weight dim 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: ParallelGroups,
        *,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.input_is_parallel = input_is_parallel

        tp_size = _tp_size(groups)
        _require_divisible(in_features, tp_size, "in_features")
        self.local_in_features = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _debug_tp(
            self.groups,
            f"row_linear_start in={self.in_features} out={self.out_features} shape={tuple(x.shape)}",
        )
        local_input = x if self.input_is_parallel else _slice_last_dim(x, self.groups)

        partial_output = F.linear(local_input, self.weight, None)
        _debug_tp(self.groups, f"row_linear_partial_done shape={tuple(partial_output.shape)}")
        output = _all_reduce_tensor_parallel_autograd(
            partial_output,
            self.groups,
        )
        _debug_tp(self.groups, f"row_linear_done shape={tuple(output.shape)}")

        if self.bias is not None:
            output = output + self.bias

        return output

    @torch.no_grad()
    def load_from_linear(self, linear: nn.Linear) -> None:
        if linear.weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"linear weight shape {tuple(linear.weight.shape)} does not match "
                f"expected {(self.out_features, self.in_features)}"
            )

        tp_rank = _tp_rank(self.groups)
        start = tp_rank * self.local_in_features
        end = start + self.local_in_features

        self.weight.copy_(linear.weight[:, start:end])

        if self.bias is not None:
            if linear.bias is None:
                raise ValueError("source linear has no bias")
            self.bias.copy_(linear.bias)
