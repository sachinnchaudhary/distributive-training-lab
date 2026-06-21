from __future__ import annotations 

from dataclasses import dataclass  
from collections.abc import Sequence

import torch
import torch.nn as nn  

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.core.distributed.placement import split_range
from meshtrain.parallelism.pipeline_parallel.p2p import (
    pipeline_prev_rank,
    pipeline_next_rank,
)

@dataclass(frozen=True)  
class PipelineStageInfo:  
    rank: int  
    stage_index: int 
    num_stages: int 
    prev_rank: int | None  
    next_rank: int | None 
    layer_start: int  
    layer_end: int  
    virtual_stage_index: int | None = None   
    global_virtual_stage_index: int | None = None
    num_virtual_stages: int | None = None
    virtual_stages_per_rank: int | None = None


@dataclass(frozen=True)  
class VirtualPipelineStageInfo:  
    rank: int  
    physical_stage_index: int  
    virtual_stage_index: int
    global_virtual_stage_index: int
    num_physical_stages: int
    virtual_stages_per_rank: int
    num_virtual_stages: int
    layer_start: int
    layer_end: int



class PipelineStage(nn.Module):  
    def __init__(self, 
                 layers: Sequence[nn.Module], 
                 groups: ParallelGroups, 
                 *, 
                 layer_start: int, 
                 layer_end: int, 
                 virtual_stage_index: int | None = None,
                 global_virtual_stage_index: int | None = None,
                 num_virtual_stages: int | None = None,
                 virtual_stages_per_rank: int | None = None,
                 ):
        
        super().__init__()  

        if layer_start < 0:
            raise ValueError("layer_start must be >= 0")
        if layer_end < layer_start:
            raise ValueError("layer_end must be >= layer_start")
        
        if len(layers) != layer_end - layer_start:
            raise ValueError("number of local layers must match layer range")
        
        self.groups = groups 
        self.layer_start = layer_start  
        self.layer_end = layer_end 

        self.layers = nn.ModuleList(layers) 
        
        self.stage_index = groups.pp_ranks.index(groups.rank) 
        self.num_stages = len(groups.pp_ranks)  

        self.prev_rank = pipeline_prev_rank(groups)
        self.next_rank = pipeline_next_rank(groups) 

        self.virtual_stage_index = virtual_stage_index
        self.global_virtual_stage_index = global_virtual_stage_index
        self.num_virtual_stages = num_virtual_stages
        self.virtual_stages_per_rank = virtual_stages_per_rank

    @property
    def is_first(self) -> bool: 
        return self.prev_rank is None  

    @property 
    def is_last(self) -> bool:  
        return self.next_rank is None
        
    def info(self) -> PipelineStageInfo: 
        return PipelineStageInfo( 
          rank=self.groups.rank,
          stage_index=self.stage_index,
          num_stages=self.num_stages,
          prev_rank=self.prev_rank,
          next_rank=self.next_rank,
          layer_start=self.layer_start,
          layer_end=self.layer_end,
          virtual_stage_index=self.virtual_stage_index,
          global_virtual_stage_index=self.global_virtual_stage_index,
          num_virtual_stages=self.num_virtual_stages,
          virtual_stages_per_rank=self.virtual_stages_per_rank,
          )

    def virtual_info(self) -> VirtualPipelineStageInfo:
        if (
            self.virtual_stage_index is None
            or self.global_virtual_stage_index is None
            or self.num_virtual_stages is None
            or self.virtual_stages_per_rank is None
        ):
            raise RuntimeError("stage is not a virtual pipeline stage")

        return VirtualPipelineStageInfo(
            rank=self.groups.rank,
            physical_stage_index=self.stage_index,
            virtual_stage_index=self.virtual_stage_index,
            global_virtual_stage_index=self.global_virtual_stage_index,
            num_physical_stages=self.num_stages,
            virtual_stages_per_rank=self.virtual_stages_per_rank,
            num_virtual_stages=self.num_virtual_stages,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )
        
    def forward_local(self, x:torch.Tensor) -> torch.Tensor: 
        for layer in self.layers: 
                x = layer(x)  
        return x  

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.forward_local(x)  

      

def build_pipeline_stage_from_layers(
        layers: Sequence[nn.Module], 
        groups: ParallelGroups, 
) -> PipelineStage:

    stage_index = groups.pp_ranks.index(groups.rank) 
    num_stages = len(groups.pp_ranks)  

    layer_range = split_range(
        size=len(layers), 
        parts=num_stages, 
        index= stage_index, 
        require_even=False,
    ) 

    local_layers = layers[layer_range.start : layer_range.end]  

    return PipelineStage( 
        local_layers, 
        groups, 
        layer_start=layer_range.start, 
        layer_end=layer_range.end, 
    )


def build_interleaved_pipeline_stages_from_layers(
        layers: Sequence[nn.Module],
        groups: ParallelGroups,
        *,
        virtual_stages_per_rank: int,
) -> list[PipelineStage]:
    if virtual_stages_per_rank < 1:
        raise ValueError("virtual_stages_per_rank must be >= 1")

    physical_stage_index = groups.pp_ranks.index(groups.rank)
    num_physical_stages = len(groups.pp_ranks)
    num_virtual_stages = num_physical_stages * virtual_stages_per_rank

    stages: list[PipelineStage] = []

    for virtual_stage_index in range(virtual_stages_per_rank):
        global_virtual_stage_index = (
            virtual_stage_index * num_physical_stages
            + physical_stage_index
        )

        layer_range = split_range(
            size=len(layers),
            parts=num_virtual_stages,
            index=global_virtual_stage_index,
            require_even=False,
        )

        local_layers = layers[layer_range.start : layer_range.end]

        stages.append(
            PipelineStage(
                local_layers,
                groups,
                layer_start=layer_range.start,
                layer_end=layer_range.end,
                virtual_stage_index=virtual_stage_index,
                global_virtual_stage_index=global_virtual_stage_index,
                num_virtual_stages=num_virtual_stages,
                virtual_stages_per_rank=virtual_stages_per_rank,
            )
        )

    return stages




