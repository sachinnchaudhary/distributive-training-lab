from __future__ import annotations 

import torch
import torch.nn as nn
import torch.nn.functional as F

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.parallelism.tensor_parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)    


def _tp_size(groups: ParallelGroups) -> int:
    return len(groups.tp_ranks)


def _tp_rank(groups: ParallelGroups) -> int:
    return groups.tp_ranks.index(groups.rank)


def _require_divisible(value: int, parts: int, name: str) -> None:
    if parts < 1:
        raise ValueError(f"parts must be at least 1, got {parts}")
    if value % parts != 0:
            raise ValueError(f"{name}={value} must be divisible by tp_size={parts}")  
    

class TensorParallelSelfAttention(nn.Module):  
    def __init__(
            self, 
            dim: int, 
            n_heads: int, 
            groups: ParallelGroups, 
            *, 
            dropout: float = 0.0, 
            bias: bool = False,  
    ): 

        super().__init__()  

        tp_size = _tp_size(groups)  

        _require_divisible(dim, n_heads, "dim")
        _require_divisible(n_heads, tp_size, "n_heads")

        self.dim = dim 
        self.n_heads = n_heads 
        self.groups = groups 
        self.dropout = dropout

        self.head_dim = dim // n_heads  
        self.local_heads = n_heads // tp_size  
        self.local_dim = self.local_heads * self.head_dim  

        self.qkv = ColumnParallelLinear(
            in_features= dim, 
            out_features=3 * dim, 
            groups = groups, 
            bias = bias, 
            gather_output= False,
        )   

        self.out_proj = RowParallelLinear(
            in_features=dim, 
            out_features=dim, 
            groups=groups, 
            bias=bias, 
            input_is_parallel=True, 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape 

        if dim != self.dim:
            raise ValueError(f"input dim {dim} does not match attention dim {self.dim}")
            
        local_qkv = self.qkv(x)
        q, k, v = local_qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.local_heads, self.head_dim)
        k = k.view(batch, seq_len, self.local_heads, self.head_dim)
        v = v.view(batch, seq_len, self.local_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, self.local_dim)

        return self.out_proj(attn)
           
    @torch.no_grad()
    def load_from_attention(self, attention: nn.Module) -> None:
        tp_rank = _tp_rank(self.groups)
        start = tp_rank * self.local_dim
        end = start + self.local_dim

        q_weight, k_weight, v_weight = attention.qkv.weight.chunk(3, dim=0)
        local_qkv_weight = torch.cat(
            [
                q_weight[start:end, :],
                k_weight[start:end, :],
                v_weight[start:end, :],
            ],
            dim=0,
        )
        self.qkv.weight.copy_(local_qkv_weight)

        if self.qkv.bias is not None:
            if attention.qkv.bias is None:
                raise ValueError("source attention qkv has no bias")

            q_bias, k_bias, v_bias = attention.qkv.bias.chunk(3, dim=0)
            local_qkv_bias = torch.cat(
                [
                    q_bias[start:end],
                    k_bias[start:end],
                    v_bias[start:end],
                ],
                dim=0,
            )
            self.qkv.bias.copy_(local_qkv_bias)

        self.out_proj.load_from_linear(attention.out_proj)
        
           


