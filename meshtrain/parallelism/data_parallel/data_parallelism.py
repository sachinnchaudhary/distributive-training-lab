from __future__ import annotations 
from dataclasses import dataclass  

import torch
import torch.nn as nn  
import torch.distributed as dist  

from meshtrain.core.distributed.groups import ParallelGroups  



@dataclass(frozen=True)  
class LossStats:  
    loss_sum: torch.Tensor  
    num_tokens: torch.Tensor  

    @property 
    def loss(self) -> torch.Tensor:  
        return self.loss_sum / self.num_tokens.clamp_min(1)    


@dataclass(frozen=True)  
class ReplicaCheck:  
    passed: bool  
    max_abs_diff: float  
    

def _dp_is_active(groups: ParallelGroups) -> bool:  
    return groups.dp_group is not None and len(groups.dp_ranks)  > 1 


def _first_parameter_device(model: nn.Module) -> torch.device:  
    for parameter in model.parameters():  
        return parameter.device  
    
    raise ValueError("model has no parameters")  



def broadcast_parameters(
        model: nn.Module, 
        groups: ParallelGroups,  
        src_rank: int | None = None, 
) -> None:  
    
    if not _dp_is_active(groups):  
        return  
    
    if src_rank is None:  
        src_rank = groups.dp_ranks[0]  
    
    for parameter in model.parameters():  
        dist.broadcast(
            parameter.data, 
            src = src_rank, 
            group= groups.dp_group,
        )
    

def sync_loss_stats(  
        loss_sum: torch.Tensor, 
        num_tokens: torch.Tensor,  
        groups: ParallelGroups,  
) -> LossStats:  
    
    global_loss_sum = loss_sum.detach().clone()  
    global_num_tokens = num_tokens.detach().clone()  

    if _dp_is_active(groups):  
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=groups.dp_group) 
        dist.all_reduce(global_num_tokens, op=dist.ReduceOp.SUM, group=groups.dp_group)

    return LossStats(
        loss_sum = global_loss_sum, 
        num_tokens=global_num_tokens, 
    )


def sync_gradients(
        model: nn.Module, 
        groups: ParallelGroups,  
) -> None:  
    
    if not _dp_is_active(groups):  
        return  
    
    for parameter in model.parameters():  
        if parameter.grad is None: 
            continue  

        dist.all_reduce(
            parameter.grad, 
            op=dist.ReduceOp.SUM, 
            group=groups.dp_group
        )
    

def check_replicas_match(
        model: nn.Module,  
        groups: ParallelGroups, 
        *, 
        atol: float = 1e-5,  
) -> ReplicaCheck:  

   if not _dp_is_active(groups):   
       return ReplicaCheck(passed=True, max_abs_diff=0.0)  

   device = _first_parameter_device(model) 
   max_diff = torch.zeros((), device=device)  

   src_rank = groups.dp_ranks[0]  

   for parameter in model.parameters():  
         reference = parameter.detach().clone()  

         dist.broadcast(
             reference, 
             src=src_rank, 
             group= groups.dp_group,
         )  

         diff = (parameter.detach() - reference).abs().max()  
         max_diff = torch.maximum(max_diff, diff)  

    
   dist.all_reduce(
       max_diff, 
       op=dist.ReduceOp.MAX, 
       group=groups.dp_group,
   )

   max_abs_diff = float(max_diff.item()) 

   return ReplicaCheck(
       passed=max_abs_diff <= atol, 
       max_abs_diff=max_abs_diff,
   )

