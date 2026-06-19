from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed as dist
import torch.nn as nn

from meshtrain.core.distributed.groups import ParallelGroups


class ZeroStage(IntEnum):
    OPTIMIZER = 1
    GRADIENT = 2
    PARAMETER = 3


@dataclass(frozen=True)
class ZeroConfig:
    stage: ZeroStage = ZeroStage.OPTIMIZER
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass(frozen=True)
class OwnedParameter:
    name: str
    parameter: nn.Parameter
    owner_rank: int


@dataclass(frozen=True)
class FlatParameterInfo:  
    name: str  
    parameter: nn.Parameter
    start: int   
    end: int  
    shape: torch.Size 
    dtype: torch.dtype  
    device: torch.device  

@dataclass(frozen=True)  
class FlatShard:  
    rank: int  
    start: int  
    end: int  

    @property
    def numel(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class FlatShardPlan:
    total_numel: int
    padded_numel: int
    chunk_numel: int
    shards: dict[int, FlatShard]

@dataclass(frozen=True)  
class ParamShardSlice:  
    name: str  
    parameter: nn.Parameter  
    param_start: int  
    param_end: int  
    shard_start: int  
    shard_end: int  


def build_flat_parameter_infos(model: nn.Module) -> list[FlatParameterInfo]:
    offset = 0  
    infos: list[FlatParameterInfo] = []

    for name, parameter in model.named_parameters():  
        if not parameter.requires_grad:  
            continue  
    
        start = offset  
        end = start + parameter.numel()  

        infos.append(
            FlatParameterInfo(
                name=name,
                parameter=parameter,
                start=start,
                end=end,
                shape=parameter.shape,
                dtype=parameter.dtype,
                device=parameter.device,
            )
        )

        offset = end  
    return infos  

def total_flat_numel(infos: list[FlatParameterInfo]) -> int:
    if not infos:
        return 0
    return infos[-1].end


def _split_range(size: int, parts: int, index: int) -> tuple[int, int]:
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if parts < 1:
        raise ValueError(f"parts must be at least 1, got {parts}")
    if index < 0 or index >= parts:
        raise ValueError(f"index must be in [0, {parts}), got {index}")

    base = size // parts
    remainder = size % parts
    start = index * base + min(index, remainder)
    length = base + (1 if index < remainder else 0)
    return start, start + length


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    return (a + b - 1) // b


def split_flat_range(total_numel: int, groups: ParallelGroups) -> dict[int, FlatShard]:
    dp_size = len(groups.dp_ranks)
    if dp_size < 1:
        raise ValueError("dp_ranks must contain at least one rank")

    shards: dict[int, FlatShard] = {}
    for rank_index, rank in enumerate(groups.dp_ranks):
        start, end = _split_range(total_numel, dp_size, rank_index)
        shards[rank] = FlatShard(rank=rank, start=start, end=end)

    return shards


def build_padded_flat_shard_plan(
    total_numel: int,
    groups: ParallelGroups,
) -> FlatShardPlan:
    dp_size = len(groups.dp_ranks)
    if dp_size < 1:
        raise ValueError("dp_ranks must contain at least one rank")

    chunk_numel = _ceil_div(total_numel, dp_size)
    padded_numel = chunk_numel * dp_size

    shards: dict[int, FlatShard] = {}
    for rank_index, rank in enumerate(groups.dp_ranks):
        start = rank_index * chunk_numel
        end = start + chunk_numel
        shards[rank] = FlatShard(rank=rank, start=start, end=end)

    return FlatShardPlan(
        total_numel=total_numel,
        padded_numel=padded_numel,
        chunk_numel=chunk_numel,
        shards=shards,
    )


def build_param_shard_slices(
    infos: list[FlatParameterInfo],
    shard: FlatShard,
) -> list[ParamShardSlice]:
    slices: list[ParamShardSlice] = []

    for info in infos:  
        overlap_start = max(info.start, shard.start)  
        overlap_end = min(info.end, shard.end)  
        
        if overlap_start >= overlap_end:  
            continue 
    
        param_start = overlap_start - info.start  
        param_end = overlap_end - info.start  

        shard_start = overlap_start - shard.start  
        shard_end = overlap_end - shard.start

        slices.append(
            ParamShardSlice(
                name=info.name,
                parameter=info.parameter,
                param_start=param_start,
                param_end=param_end,
                shard_start=shard_start,
                shard_end=shard_end,
            )
        )

    return slices


def flatten_parameters(infos: list[FlatParameterInfo]) -> torch.Tensor:
    if not infos:
        raise ValueError("cannot flatten an empty parameter list")

    return torch.cat(
        [info.parameter.detach().reshape(-1) for info in infos],
        dim=0,
    )


def flatten_gradients(infos: list[FlatParameterInfo]) -> torch.Tensor:
    if not infos:
        raise ValueError("cannot flatten gradients for an empty parameter list")

    pieces = []
    for info in infos:
        if info.parameter.grad is None:
            raise RuntimeError(f"missing gradient for parameter {info.name}")
        pieces.append(info.parameter.grad.detach().reshape(-1))

    return torch.cat(pieces, dim=0)


def pad_flat_tensor(flat: torch.Tensor, padded_numel: int) -> torch.Tensor:
    if padded_numel < flat.numel():
        raise ValueError(
            f"padded_numel {padded_numel} must be >= flat.numel() {flat.numel()}"
        )
    if padded_numel == flat.numel():
        return flat

    padded = torch.zeros(
        padded_numel,
        dtype=flat.dtype,
        device=flat.device,
    )
    padded[: flat.numel()].copy_(flat)
    return padded


def chunk_flat_tensor(flat: torch.Tensor, chunk_numel: int) -> list[torch.Tensor]:
    if chunk_numel <= 0:
        raise ValueError(f"chunk_numel must be positive, got {chunk_numel}")
    if flat.numel() % chunk_numel != 0:
        raise ValueError(
            f"flat tensor with {flat.numel()} elements is not divisible by chunk_numel={chunk_numel}"
        )
    return list(flat.split(chunk_numel))


def copy_flat_to_parameters(
    flat: torch.Tensor,
    infos: list[FlatParameterInfo],
) -> None:
    expected_numel = total_flat_numel(infos)
    if flat.numel() != expected_numel:
        raise ValueError(
            f"flat tensor has {flat.numel()} elements, expected {expected_numel}"
        )

    with torch.no_grad():
        for info in infos:
            piece = flat[info.start : info.end].view(info.shape)
            info.parameter.copy_(piece)

def _dp_is_active(groups: ParallelGroups) -> bool:
    return groups.dp_group is not None and len(groups.dp_ranks) > 1


def build_owned_parameters(
    model: nn.Module,
    groups: ParallelGroups,
) -> list[OwnedParameter]:
    dp_size = len(groups.dp_ranks)
    if dp_size < 1:
        raise ValueError("dp_ranks must contain at least one rank")

    owned: list[OwnedParameter] = []
    trainable_index = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        owner_rank = groups.dp_ranks[trainable_index % dp_size]
        owned.append(
            OwnedParameter(
                name=name,
                parameter=parameter,
                owner_rank=owner_rank,
            )
        )
        trainable_index += 1

    return owned


class ZeroAdamW:
    def __init__(
        self,
        model: nn.Module,
        groups: ParallelGroups,
        config: ZeroConfig | None = None,
    ):
        self.model = model
        self.groups = groups
        self.config = config or ZeroConfig()
        self.config = ZeroConfig(
            stage=ZeroStage(self.config.stage),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        self.flat_infos = build_flat_parameter_infos(model)
        self.total_numel = total_flat_numel(self.flat_infos)
        self.flat_shard_plan = build_padded_flat_shard_plan(self.total_numel, groups)
        self.local_flat_shard = self.flat_shard_plan.shards[groups.rank]
        self.local_grad_shard: torch.Tensor | None = None
        self.local_param_shard: torch.Tensor | None = None
        self.owned_parameters = build_owned_parameters(model, groups)
        self.state: dict[str, dict[str, torch.Tensor]] = {}
        self.step_count = 0

        if self.config.stage == ZeroStage.OPTIMIZER:
            for owned_param in self.owned_parameters:
                if self._owns(owned_param):
                    parameter = owned_param.parameter
                    self.state[owned_param.name] = {
                        "exp_avg": torch.zeros_like(parameter),
                        "exp_avg_sq": torch.zeros_like(parameter),
                    }
        elif self.config.stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER):
            self.state["flat"] = {
                "exp_avg": torch.zeros(
                    self.local_flat_shard.numel,
                    dtype=self.flat_infos[0].dtype,
                    device=self.flat_infos[0].device,
                ),
                "exp_avg_sq": torch.zeros(
                    self.local_flat_shard.numel,
                    dtype=self.flat_infos[0].dtype,
                    device=self.flat_infos[0].device,
                ),
            }

            if self.config.stage == ZeroStage.PARAMETER:
                flat_params = flatten_parameters(self.flat_infos)
                padded_params = pad_flat_tensor(
                    flat_params,
                    self.flat_shard_plan.padded_numel,
                )
                self.local_param_shard = padded_params[
                    self.local_flat_shard.start : self.local_flat_shard.end
                ].clone()

    def _owns(self, owned_param: OwnedParameter) -> bool:
        return owned_param.owner_rank == self.groups.rank

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.detach_()
                parameter.grad.zero_()

    def sync_gradients(self) -> None:
        if self.config.stage == ZeroStage.OPTIMIZER:
            self._sync_full_gradients()
            return

        if self.config.stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER):
            self._sync_flat_gradient_shard()
            return

        raise NotImplementedError("ZeRO-3 gradient synchronization is not implemented yet")

    def _sync_full_gradients(self) -> None:
        if not _dp_is_active(self.groups):
            return

        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
                group=self.groups.dp_group,
            )

    def _sync_flat_gradient_shard(self) -> None:
        flat_grad = flatten_gradients(self.flat_infos)
        padded_grad = pad_flat_tensor(
            flat_grad,
            self.flat_shard_plan.padded_numel,
        )
        chunks = chunk_flat_tensor(
            padded_grad,
            self.flat_shard_plan.chunk_numel,
        )

        local_grad_shard = torch.empty(
            self.flat_shard_plan.chunk_numel,
            dtype=padded_grad.dtype,
            device=padded_grad.device,
        )

        if _dp_is_active(self.groups):
            dist.reduce_scatter(
                local_grad_shard,
                chunks,
                op=dist.ReduceOp.SUM,
                group=self.groups.dp_group,
            )
        else:
            local_grad_shard.copy_(chunks[0])

        self.local_grad_shard = local_grad_shard
        self._clear_full_gradients()

    def _clear_full_gradients(self) -> None:
        for parameter in self.model.parameters():
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        if self.config.stage == ZeroStage.GRADIENT:
            self._step_flat_sharded()
            return
        if self.config.stage == ZeroStage.PARAMETER:
            self._step_parameter_sharded()
            return

        self.step_count += 1

        for owned_param in self.owned_parameters:
            if not self._owns(owned_param):
                continue

            parameter = owned_param.parameter
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for owned parameter {owned_param.name}")

            self._adamw_update(owned_param.name, parameter, parameter.grad)

        self._broadcast_updated_parameters()

    def _step_flat_sharded(self) -> None:
        if self.local_grad_shard is None:
            raise RuntimeError("call sync_gradients() before ZeRO-2 step()")

        self.step_count += 1

        flat_params = flatten_parameters(self.flat_infos)
        padded_params = pad_flat_tensor(
            flat_params,
            self.flat_shard_plan.padded_numel,
        )

        local_param_shard = padded_params[
            self.local_flat_shard.start : self.local_flat_shard.end
        ].clone()

        state = self.state["flat"]
        self._adamw_update_tensor(
            local_param_shard,
            self.local_grad_shard,
            state["exp_avg"],
            state["exp_avg_sq"],
        )

        gathered_shards = [
            torch.empty_like(local_param_shard)
            for _ in self.groups.dp_ranks
        ]

        if _dp_is_active(self.groups):
            dist.all_gather(
                gathered_shards,
                local_param_shard,
                group=self.groups.dp_group,
            )
        else:
            gathered_shards[0].copy_(local_param_shard)

        full_padded_params = torch.cat(gathered_shards, dim=0)
        full_params = full_padded_params[: self.total_numel]
        copy_flat_to_parameters(full_params, self.flat_infos)

        self.local_grad_shard = None

    def _step_parameter_sharded(self) -> None:
        if self.local_param_shard is None:
            raise RuntimeError("ZeRO-3 local_param_shard has not been initialized")
        if self.local_grad_shard is None:
            raise RuntimeError("call sync_gradients() before ZeRO-3 step()")

        self.step_count += 1

        state = self.state["flat"]
        self._adamw_update_tensor(
            self.local_param_shard,
            self.local_grad_shard,
            state["exp_avg"],
            state["exp_avg_sq"],
        )

        self.local_grad_shard = None
        self.gather_parameters()

    def gather_parameters(self) -> None:
        if self.config.stage != ZeroStage.PARAMETER:
            return
        if self.local_param_shard is None:
            raise RuntimeError("ZeRO-3 local_param_shard has not been initialized")

        gathered_shards = [
            torch.empty_like(self.local_param_shard)
            for _ in self.groups.dp_ranks
        ]

        if _dp_is_active(self.groups):
            dist.all_gather(
                gathered_shards,
                self.local_param_shard,
                group=self.groups.dp_group,
            )
        else:
            gathered_shards[0].copy_(self.local_param_shard)

        full_padded_params = torch.cat(gathered_shards, dim=0)
        full_params = full_padded_params[: self.total_numel]
        copy_flat_to_parameters(full_params, self.flat_infos)

    def _adamw_update(
        self,
        name: str,
        parameter: nn.Parameter,
        grad: torch.Tensor,
    ) -> None:
        beta1, beta2 = self.config.betas
        state = self.state[name]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        if self.config.weight_decay != 0.0:
            parameter.mul_(1.0 - self.config.lr * self.config.weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count
        step_size = self.config.lr / bias_correction1
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.config.eps)

        parameter.addcdiv_(exp_avg, denom, value=-step_size)

    def _adamw_update_tensor(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
    ) -> None:
        beta1, beta2 = self.config.betas

        if self.config.weight_decay != 0.0:
            parameter.mul_(1.0 - self.config.lr * self.config.weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count
        step_size = self.config.lr / bias_correction1
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.config.eps)

        parameter.addcdiv_(exp_avg, denom, value=-step_size)

    def _broadcast_updated_parameters(self) -> None:
        if not _dp_is_active(self.groups):
            return

        for owned_param in self.owned_parameters:
            dist.broadcast(
                owned_param.parameter.data,
                src=owned_param.owner_rank,
                group=self.groups.dp_group,
            )

    def local_state_numel(self) -> int:
        total = 0
        for state in self.state.values():
            total += state["exp_avg"].numel()
            total += state["exp_avg_sq"].numel()
        return total

    def local_owned_parameter_names(self) -> list[str]:
        return [
            owned_param.name
            for owned_param in self.owned_parameters
            if self._owns(owned_param)
        ]

    def local_grad_shard_numel(self) -> int:
        if self.local_grad_shard is None:
            return 0
        return self.local_grad_shard.numel()

    def local_param_shard_numel(self) -> int:
        if self.local_param_shard is None:
            return 0
        return self.local_param_shard.numel()
