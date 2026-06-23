from __future__ import annotations  

import torch  
import torch.distributed as dist  

from meshtrain.core.distributed.groups import ParallelGroups  



def _pp_is_active(groups: ParallelGroups) -> bool:  
    return groups.pp_group is not None and len(groups.pp_ranks) > 1  


def _pp_group(groups: ParallelGroups) -> dist.ProcessGroup | None:
    return groups.pp_group if _pp_is_active(groups) else None


def _pp_group_rank(groups: ParallelGroups, rank: int) -> int:
    return groups.pp_ranks.index(rank)



def pipeline_prev_rank(groups: ParallelGroups) -> int | None:  

    pp_index = groups.pp_ranks.index(groups.rank)  
    if pp_index == 0: 
        return None 

    return groups.pp_ranks[pp_index - 1]  


def pipeline_next_rank(groups: ParallelGroups) -> int | None:  
    pp_index = groups.pp_ranks.index(groups.rank)  

    if pp_index == len(groups.pp_ranks) - 1:  
        return None  
    
    return groups.pp_ranks[pp_index + 1]  




def is_first_pipeline_stage(groups: ParallelGroups) -> bool:
    return pipeline_prev_rank(groups) is None  


def is_last_pipeline_stage(groups: ParallelGroups) -> bool:
    return pipeline_next_rank(groups) is None




def send_forward(tensor: torch.Tensor, groups: ParallelGroups) -> None:  
    dst = pipeline_next_rank(groups) 

    if dst is None:  
        return  
    
    work = dist.isend(
        tensor.contiguous(),
        group=_pp_group(groups),
        group_dst=_pp_group_rank(groups, dst),
    )  
    work.wait()

def recv_forward(
        shape: tuple[int, ...], 
        groups: ParallelGroups, 
        *, 
        device: torch.device, 
        dtype: torch.dtype, 
) -> torch.Tensor:  
    
    src = pipeline_prev_rank(groups)  

    if src is None: 
        raise RuntimeError("first pipeline stage cannot receive forward activation") 
    
    tensor = torch.empty(shape, device=device, dtype=dtype)  
    work = dist.irecv(
        tensor,
        group=_pp_group(groups),
        group_src=_pp_group_rank(groups, src),
    )  
    work.wait()

    return tensor  




def send_backward(tensor_grad: torch.Tensor, groups: ParallelGroups) -> None:  
    dst = pipeline_prev_rank(groups) 

    if dst is None:  
        return  
    
    work = dist.isend(
        tensor_grad.contiguous(),
        group=_pp_group(groups),
        group_dst=_pp_group_rank(groups, dst),
    ) 
    work.wait()

def recv_backward( 
    shape: tuple[int, ...], 
    groups: ParallelGroups, 
    *, 
    device: torch.device, 
    dtype: torch.dtype,  
) -> torch.Tensor:  
    
    src = pipeline_next_rank(groups)  
    
    if src is None: 
        raise RuntimeError("last pipeline stage cannot receive backward activation gradient")  
    
    tensor_grad = torch.empty(shape, device=device, dtype=dtype)  
    work = dist.irecv(
        tensor_grad,
        group=_pp_group(groups),
        group_src=_pp_group_rank(groups, src),
    )  
    work.wait()

    return tensor_grad  


def virtual_stage_owner_rank(
        groups: ParallelGroups,  
        global_virtual_stage_index: int, 
) -> int:  

    pp_size = len(groups.pp_ranks)  
    if pp_size < 1:
        raise ValueError("pp_ranks must contain at least one rank")

    physical_stage_index = global_virtual_stage_index % pp_size  
    return groups.pp_ranks[physical_stage_index]  
  

def virtual_forward_dst_rank(
   groups: ParallelGroups,  
   *,  
   global_virtual_stage_index: int,  
   num_virtual_stages: int,         
) -> int | None:  
    
    next_virtual_stage_index = global_virtual_stage_index + 1

    if next_virtual_stage_index >= num_virtual_stages:
        return None

    return virtual_stage_owner_rank(groups, next_virtual_stage_index)


def virtual_forward_src_rank(
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,      
) -> int | None:  
    
    previous_virtual_stage_index = global_virtual_stage_index - 1

    if previous_virtual_stage_index < 0:
        return None

    return virtual_stage_owner_rank(groups, previous_virtual_stage_index)  

def virtual_backward_dst_rank(
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
) -> int | None:
    previous_virtual_stage_index = global_virtual_stage_index - 1

    if previous_virtual_stage_index < 0:
        return None

    return virtual_stage_owner_rank(groups, previous_virtual_stage_index)  


def virtual_backward_src_rank(
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
    num_virtual_stages: int,
) -> int | None:
    next_virtual_stage_index = global_virtual_stage_index + 1

    if next_virtual_stage_index >= num_virtual_stages:
        return None

    return virtual_stage_owner_rank(groups, next_virtual_stage_index)  



def send_virtual_forward(
    tensor: torch.Tensor,
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
    num_virtual_stages: int,
) -> None:
    dst = virtual_forward_dst_rank(
        groups,
        global_virtual_stage_index=global_virtual_stage_index,
        num_virtual_stages=num_virtual_stages,
    )

    if dst is None:
        return

    work = dist.isend(
        tensor.contiguous(),
        group=_pp_group(groups),
        group_dst=_pp_group_rank(groups, dst),
    )
    work.wait()


def recv_virtual_forward(
    shape: tuple[int, ...],
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    src = virtual_forward_src_rank(
        groups,
        global_virtual_stage_index=global_virtual_stage_index,
    )

    if src is None:
        raise RuntimeError(
            "first virtual pipeline stage cannot receive forward activation"
        )

    tensor = torch.empty(shape, device=device, dtype=dtype)
    work = dist.irecv(
        tensor,
        group=_pp_group(groups),
        group_src=_pp_group_rank(groups, src),
    )
    work.wait()

    return tensor  


def send_virtual_backward(
    tensor_grad: torch.Tensor,
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
) -> None:
    dst = virtual_backward_dst_rank(
        groups,
        global_virtual_stage_index=global_virtual_stage_index,
    )

    if dst is None:
        return

    work = dist.isend(
        tensor_grad.contiguous(),
        group=_pp_group(groups),
        group_dst=_pp_group_rank(groups, dst),
    )
    work.wait()


def recv_virtual_backward(
    shape: tuple[int, ...],
    groups: ParallelGroups,
    *,
    global_virtual_stage_index: int,
    num_virtual_stages: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    src = virtual_backward_src_rank(
        groups,
        global_virtual_stage_index=global_virtual_stage_index,
        num_virtual_stages=num_virtual_stages,
    )

    if src is None:
        raise RuntimeError(
            "last virtual pipeline stage cannot receive backward activation gradient"
        )

    tensor_grad = torch.empty(shape, device=device, dtype=dtype)
    work = dist.irecv(
        tensor_grad,
        group=_pp_group(groups),
        group_src=_pp_group_rank(groups, src),
    )
    work.wait()

    return tensor_grad
