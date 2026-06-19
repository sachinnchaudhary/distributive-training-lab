from __future__ import annotations  
from dataclasses import dataclass  

import torch 
import torch.nn as nn  
import torch.distributed as dist  

from meshtrain.core.distributed.groups import ParallelGroups 
from meshtrain.parallelism.data_parallel.zero import (
    build_flat_parameter_infos,
    build_padded_flat_shard_plan,
    copy_flat_to_parameters,
    flatten_parameters,
    pad_flat_tensor,
    total_flat_numel,
) 

@dataclass(frozen=True)  
class FSDPConfig:  
    reshard_after_forward: bool = True  


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

    def forward(self, *args, **kwargs):
        self.gather_full_parameters()
        output = self.module(*args, **kwargs)

        if self.config.reshard_after_forward:
            self.reshard_parameters()

        return output

