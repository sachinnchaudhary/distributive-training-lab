from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.parallelism.tensor_parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)  


class TensorParallelMLP(nn.Module):  
    def __init__(
            self, 
            dim: int, 
            hidden_dim: int,  
            groups: ParallelGroups, 
            *, 
            bias: bool = False
    ):  
    
      super().__init__()  
      
      self.gate_proj = ColumnParallelLinear(
         dim, 
         hidden_dim, 
         groups, 
         bias=bias, 
         gather_output=False,
      )

      self.up_proj = ColumnParallelLinear(
         dim, 
         hidden_dim, 
         groups, 
         bias=bias, 
         gather_output=False,
      )
      self.down_proj = RowParallelLinear(
         hidden_dim, 
         dim, 
         groups, 
         bias=bias, 
         input_is_parallel=True,
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)

    @torch.no_grad()
    def load_from_mlp(self, mlp: nn.Module) -> None:
        self.gate_proj.load_from_linear(mlp.gate_proj)
        self.up_proj.load_from_linear(mlp.up_proj)
        self.down_proj.load_from_linear(mlp.down_proj)

