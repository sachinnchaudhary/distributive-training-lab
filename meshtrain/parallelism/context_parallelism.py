from __future__ import annotations  

from dataclasses import dataclass  
import math  

import torch  
import torch.distributed as dist  

from meshtrain.core.distributed.groups import ParallelGroups  
from meshtrain.core.distributed.placement import split_range  



@dataclass(frozen=True)  
class ContextShard:  
    start: int  
    end: int  
    sequence_length: int  
    cp_rank: int  
    cp_size: int  

    @property
    def length(self) -> int:  
        return self.end - self.start  
    

def _cp_size(groups: ParallelGroups) -> int:
    return len(groups.cp_ranks)  

def _cp_rank(groups: ParallelGroups) -> int:
    return groups.cp_ranks.index(groups.rank)  

def _cp_is_active(groups: ParallelGroups) -> bool:
    return groups.cp_group is not None and len(groups.cp_ranks) > 1  

def _normalize_dim(dim: int, ndim: int) -> int:  
    if ndim == 0:
        raise ValueError("cannot index a scalar tensor")

    if dim < 0:  
        dim = ndim + dim  

    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim must be in [0, {ndim}), got {dim}")

    return dim  


def context_parallel_range(
    sequence_length: int,
    groups: ParallelGroups,
) -> ContextShard:  
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    
    cp_size = _cp_size(groups)  
    cp_rank = _cp_rank(groups)  

    index_range = split_range(  
        size=sequence_length, 
        parts= cp_size, 
        index=cp_rank,  
        require_even=False,  
    )

    return ContextShard(  
        start=index_range.start,
        end=index_range.end,
        sequence_length=sequence_length,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

def shard_sequence(
        tensor: torch.Tensor,  
        groups: ParallelGroups,  
        *,  
        seq_dim: int,  
) -> torch.Tensor:  
    
    seq_dim = _normalize_dim(seq_dim, tensor.ndim)
    
    shard =  context_parallel_range(
        sequence_length=tensor.shape[seq_dim],  
        groups=groups,  
    )

    return tensor.narrow(seq_dim, shard.start, shard.length)


shard_sequnce = shard_sequence

def gather_sequence(  
    local_tensor: torch.Tensor,  
    groups: ParallelGroups,  
    *,  
    seq_dim: int,  
) -> torch.Tensor:  
    
    seq_dim = _normalize_dim(seq_dim, local_tensor.ndim)  

    if not _cp_is_active(groups):  
        return local_tensor  
    
    gathered = [  
        torch.empty_like(local_tensor)  
        for _ in range(_cp_size(groups))  
    ]

    dist.all_gather(  
        gathered,  
        local_tensor,  
        group=groups.cp_group,  
    )

    return torch.cat(gathered, dim=seq_dim)  



def global_positions_for_shard(
    shard: ContextShard, 
    *, 
    device: torch.device,  
) -> torch.Tensor:  
    
    return torch.arange( 
        shard.start,  
        shard.end,  
        device=device,  
        dtype=torch.long,  
    )

def causal_mask_for_ranges(
    query_start: int,  
    query_end: int,  
    key_start: int,  
    key_end: int, 
    *,  
    device: torch.device,   
) -> torch.Tensor:  
    
   q_positions = torch.arange(query_start, query_end, device=device)   
   k_positions = torch.arange(key_start, key_end, device=device)  

   return k_positions.unsqueeze(0) <=  q_positions.unsqueeze(1)


def _ring_next_rank(groups: ParallelGroups) -> int:    
    cp_rank = _cp_rank(groups)  
    cp_size = _cp_size(groups)  
    next_cp_rank = (cp_rank + 1) % cp_size  
    return groups.cp_ranks[next_cp_rank]  


def _ring_prev_rank(groups: ParallelGroups) -> int:  

    cp_rank = _cp_rank(groups)  
    cp_size = _cp_size(groups)  
    prev_cp_rank = (cp_rank - 1) % cp_size  
    return groups.cp_ranks[prev_cp_rank]


def _exchange_kv_block(
    k_block: torch.Tensor, 
    v_block: torch.Tensor, 
    groups: ParallelGroups, 
    )  -> tuple[torch.Tensor, torch.Tensor]: 

     if not _cp_is_active(groups):
        return k_block, v_block  

     send_rank = _ring_next_rank(groups)  
     recv_rank = _ring_prev_rank(groups)  

     recv_k = torch.empty_like(k_block) 
     recv_v = torch.empty_like(v_block)  

     ops = [  
         dist.P2POp(dist.isend, k_block, send_rank),  
         dist.P2POp(dist.isend, v_block, send_rank), 
         dist.P2POp(dist.irecv, recv_k, recv_rank), 
         dist.P2POp(dist.irecv, recv_v, recv_rank), 
     ]       
     
     works = dist.batch_isend_irecv(ops)  

     for work in works:  
         work.wait()  

     return recv_k, recv_v  


def _online_attention_update(
    q_local: torch.Tensor,  
    k_block: torch.Tensor,  
    v_block: torch.Tensor, 
    *,  
    mask: torch.Tensor,  
    scale: float,  
    running_max: torch.Tensor,  
    running_sum: torch.Tensor,  
    running_out: torch.Tensor, 
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
    
    scores = torch.matmul(
        q_local,
        k_block.transpose(-1, -2),
    ) * scale

    if mask.shape != scores.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"score shape {tuple(scores.shape[-2:])}"
        )
    
    scores = scores.masked_fill(
        ~mask.view(1, 1, mask.shape[0], mask.shape[1]),
        float("-inf"),
    )

    block_max = scores.max(dim=-1).values
    valid_block = torch.isfinite(block_max)
    safe_block_max = torch.where(
        valid_block,
        block_max,
        torch.zeros_like(block_max),
    )

    scores_exp = torch.exp(scores - safe_block_max.unsqueeze(-1))
    scores_exp = torch.where(
        torch.isfinite(scores),
        scores_exp,
        torch.zeros_like(scores_exp),
    )

    block_sum = scores_exp.sum(dim=-1)
    block_out = torch.matmul(scores_exp, v_block)

    new_max = torch.maximum(running_max, block_max)

    old_scale = torch.exp(running_max - new_max)
    old_scale = torch.where(
        torch.isfinite(running_max),
        old_scale,
        torch.zeros_like(old_scale),
    )

    block_scale = torch.exp(block_max - new_max)
    block_scale = torch.where(
        valid_block,
        block_scale,
        torch.zeros_like(block_scale),
    )

    new_sum = running_sum * old_scale + block_sum * block_scale
    new_out = (
        running_out * old_scale.unsqueeze(-1)
        + block_out * block_scale.unsqueeze(-1)
    )

    return new_max, new_sum, new_out


def ring_causal_attention(
    q_local: torch.Tensor,  
    k_local: torch.Tensor,  
    v_local: torch.Tensor,  
    groups: ParallelGroups,  
    *,  
    sequence_length: int, 
    scale: float | None = None,  
) -> torch.Tensor:  
    if q_local.ndim != 4:
        raise ValueError("q_local must have shape [B, H, T_local, D_head]")
    if k_local.shape != q_local.shape:
        raise ValueError("k_local must have the same shape as q_local")
    if v_local.shape != q_local.shape:
        raise ValueError("v_local must have the same shape as q_local")

    batch, n_heads, local_seq_len, head_dim = q_local.shape 
    cp_size = _cp_size(groups)

    if sequence_length % cp_size != 0:
        raise ValueError("ring_causal_attention currently requires even CP sequence shards")

    if scale is None:  
        scale = 1.0 / math.sqrt(head_dim)  
    
    query_shard = context_parallel_range(  
        sequence_length=sequence_length,  
        groups=groups,  
    )

    if query_shard.length != local_seq_len:  
         raise ValueError(
            f"local sequence length {local_seq_len} does not match "
            f"context shard length {query_shard.length}"
        )  
    
    running_max = torch.full(  
      (batch, n_heads, local_seq_len),  
      float("-inf"),  
      device = q_local.device,  
      dtype=q_local.dtype,  
    )

    running_sum = torch.zeros( 
        (batch, n_heads, local_seq_len),
        device=q_local.device,
        dtype=q_local.dtype,   
    )

    running_out = torch.zeros(
        (batch, n_heads, local_seq_len, head_dim),
        device=q_local.device,
        dtype=q_local.dtype,
    )

    current_k = k_local 
    current_v = v_local  

    current_owner_cp_rank = query_shard.cp_rank  

    for step in range(query_shard.cp_size):  
        key_range = split_range( 
            size=sequence_length,  
            parts=query_shard.cp_size,  
            index=current_owner_cp_rank,  
            require_even=False,  
        ) 

        mask = causal_mask_for_ranges(
            query_start=query_shard.start,
            query_end=query_shard.end,
            key_start=key_range.start,
            key_end=key_range.end,
            device=q_local.device,
        )

        running_max, running_sum, running_out = _online_attention_update(
             q_local,
            current_k,
            current_v,
            mask=mask,
            scale=scale,
            running_max=running_max,
            running_sum=running_sum,
            running_out=running_out,
        )

        if step != query_shard.cp_size - 1:  
          current_k, current_v = _exchange_kv_block(
                current_k,
                current_v,
                groups,)
          current_owner_cp_rank = (
                current_owner_cp_rank - 1
            ) % query_shard.cp_size

    return running_out / running_sum.clamp_min(torch.finfo(running_sum.dtype).tiny).unsqueeze(-1) 
