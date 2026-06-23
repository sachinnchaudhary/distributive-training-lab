from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.parallelism.tensor_parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)  


def _debug_tp(groups: ParallelGroups, message: str) -> None:
    if os.environ.get("MESHTRAIN_TP_DEBUG", "0") != "1":
        return

    if os.environ.get("MESHTRAIN_TP_SYNC_DEBUG", "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"rank={groups.rank} tp_group={groups.tp_ranks} tp_mlp:{message}",
        flush=True,
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
        _debug_tp(self.gate_proj.groups, f"forward_start shape={tuple(x.shape)}")
        _debug_tp(self.gate_proj.groups, "gate_proj_start")
        gate = self.gate_proj(x)
        _debug_tp(self.gate_proj.groups, f"gate_proj_done shape={tuple(gate.shape)}")
        _debug_tp(self.up_proj.groups, "up_proj_start")
        up = self.up_proj(x)
        _debug_tp(self.up_proj.groups, f"up_proj_done shape={tuple(up.shape)}")
        _debug_tp(self.gate_proj.groups, "activation_start")
        hidden = F.silu(gate) * up
        _debug_tp(self.gate_proj.groups, f"activation_done shape={tuple(hidden.shape)}")
        _debug_tp(self.down_proj.groups, "down_proj_start")
        output = self.down_proj(hidden)
        _debug_tp(self.down_proj.groups, f"down_proj_done shape={tuple(output.shape)}")
        return output

    @torch.no_grad()
    def load_from_mlp(self, mlp: nn.Module) -> None:
        self.gate_proj.load_from_linear(mlp.gate_proj)
        self.up_proj.load_from_linear(mlp.up_proj)
        self.down_proj.load_from_linear(mlp.down_proj)

