from __future__ import annotations  
import math
from dataclasses import dataclass  

import torch 
import torch.nn as nn  
import torch.distributed as dist  

from meshtrain.core.distributed.groups import ParallelGroups 
from meshtrain.parallelism.data_parallel.zero import (
    build_flat_parameter_infos,
    build_padded_flat_shard_plan,
    chunk_flat_tensor,
    copy_flat_to_parameters,
    flatten_gradients,
    flatten_parameters,
    pad_flat_tensor,
    total_flat_numel,
) 

@dataclass(frozen=True)  
class FSDPConfig:  
    reshard_after_forward: bool = True  
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


def _dp_is_active(groups):
    return groups.dp_group is not None and len(groups.dp_ranks) > 1  



class FullyShardedModule(nn.Module):  
    def __init__(self, module, groups, config=None):
        super().__init__()

        self.module = module  
        self.groups = groups  
        self.config = config or FSDPConfig()  

        self.flat_infos = build_flat_parameter_infos(module)  
        self.total_numel = total_flat_numel(self.flat_infos)  
        if self.total_numel == 0:
            raise ValueError("FullyShardedModule requires at least one trainable parameter")

        self.shard_plan = build_padded_flat_shard_plan(self.total_numel, groups)
        self.local_shard = self.shard_plan.shards[groups.rank]

        flat_params = flatten_parameters(self.flat_infos)
        padded_params = pad_flat_tensor(flat_params, self.shard_plan.padded_numel)

        self.local_param_shard = padded_params[
            self.local_shard.start : self.local_shard.end
        ].clone()
        self.local_grad_shard: torch.Tensor | None = None
        self.step_count = 0
        self.exp_avg = torch.zeros_like(self.local_param_shard)
        self.exp_avg_sq = torch.zeros_like(self.local_param_shard)

    def gather_full_parameters(self) -> None:
        gathered = [
            torch.empty_like(self.local_param_shard)
            for _ in self.groups.dp_ranks
        ]

        if _dp_is_active(self.groups):
            dist.all_gather(
                gathered,
                self.local_param_shard,
                group=self.groups.dp_group,
            )
        else:
            gathered[0].copy_(self.local_param_shard)

        full_padded = torch.cat(gathered, dim=0)
        full = full_padded[: self.total_numel]
        copy_flat_to_parameters(full, self.flat_infos)

    def reshard_parameters(self) -> None:
        flat_params = flatten_parameters(self.flat_infos)
        padded_params = pad_flat_tensor(flat_params, self.shard_plan.padded_numel)
        self.local_param_shard.copy_(
            padded_params[self.local_shard.start : self.local_shard.end]
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue

            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.detach_()
                parameter.grad.zero_()

    def sync_gradients(self) -> None:
        flat_grad = flatten_gradients(self.flat_infos)
        padded_grad = pad_flat_tensor(flat_grad, self.shard_plan.padded_numel)
        chunks = chunk_flat_tensor(padded_grad, self.shard_plan.chunk_numel)

        local_grad_shard = torch.empty(
            self.shard_plan.chunk_numel,
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
        for parameter in self.module.parameters():
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        if self.local_grad_shard is None:
            raise RuntimeError("call sync_gradients() before FSDP step()")

        self.step_count += 1
        self._adamw_update_local_shard()
        self.local_grad_shard = None
        self.gather_full_parameters()

    def _adamw_update_local_shard(self) -> None:
        assert self.local_grad_shard is not None
        beta1, beta2 = self.config.betas

        if self.config.weight_decay != 0.0:
            self.local_param_shard.mul_(1.0 - self.config.lr * self.config.weight_decay)

        self.exp_avg.mul_(beta1).add_(self.local_grad_shard, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(
            self.local_grad_shard,
            self.local_grad_shard,
            value=1.0 - beta2,
        )

        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count
        step_size = self.config.lr / bias_correction1
        denom = self.exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(
            self.config.eps
        )

        self.local_param_shard.addcdiv_(self.exp_avg, denom, value=-step_size)

    def local_grad_shard_numel(self) -> int:
        if self.local_grad_shard is None:
            return 0
        return self.local_grad_shard.numel()

    def local_state_numel(self) -> int:
        return self.exp_avg.numel() + self.exp_avg_sq.numel()

    def forward(self, *args, **kwargs):
        self.gather_full_parameters()
        output = self.module(*args, **kwargs)

        if self.config.reshard_after_forward:
            self.reshard_parameters()

        return output

