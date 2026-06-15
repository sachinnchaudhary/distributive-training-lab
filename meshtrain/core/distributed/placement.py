from __future__ import annotations  
from dataclasses import dataclass  
from .mesh import ParallelDims, RankMesh, RankCoord 
from .runtime import init_runtime, shutdown_runtime
from .groups import build_parallel_groups



@dataclass(frozen=True)  
class IndexRange:  
    start: int  
    end: int  
    
    @property
    def size(self) -> int:  
        return self.end - self.start  


@dataclass(frozen=True)  
class TensorShard:  
    dim: int  
    index_range: IndexRange  
    global_size: int 
    num_shards: int  
    shard_index: int 

@dataclass(frozen=True)  
class LayerPlacement:  
    layers: IndexRange  
    stage_index: int  
    num_stages: int  

@dataclass(frozen=True) 
class RankPlacement:  
    rank: int  
    coord: RankCoord
    layers: LayerPlacement
    batch: IndexRange
    context: IndexRange
    experts: IndexRange | None
    tensor: TensorShard | None


def split_range(size: int, parts: int, index: int, *, require_even: bool = False) -> IndexRange: 

    if size < 1:  
        raise ValueError(f"size must be at least 1, got {size}")  
    if parts < 1:  
        raise ValueError(f"parts must be at least 1, got {parts}")  
    if index < 0 or index >= parts:  
        raise ValueError(f"index must be in [0, {parts}), got {index}")
    
    if require_even and size % parts != 0:  
        raise ValueError(f"size {size} must divide evenly into {parts} parts") 
    
    base = size // parts  
    remainder = size % parts  

    start = index * base + min(index, remainder)  
    length = base + (1 if index < remainder else 0)  
    end = start + length  

    return IndexRange(start, end)  


def pipeline_layers(num_layers: int, coord:RankCoord, pp_size: int) -> LayerPlacement:  

    layers = split_range(
        size=num_layers, 
        parts=pp_size, 
        index=coord.pp, 
        require_even=False
    )

    return LayerPlacement(
        layers=layers, 
        stage_index=coord.pp, 
        num_stages=pp_size,
    )


def data_batch_range(global_batch_size: int, coord:RankCoord, dp_size: int) -> IndexRange:   

    return split_range(
        size=global_batch_size, 
        parts=dp_size,
        index=coord.dp, 
        require_even=False, 
    )


def context_range(sequence_length: int, coord: RankCoord, cp_size: int) -> IndexRange:  
    return split_range(
        size=sequence_length, 
        parts=cp_size, 
        index=coord.cp,  
        require_even=False,  
    )

def expert_range(num_experts: int, coord: RankCoord, ep_size: int) -> IndexRange:  
    return split_range(
        size=num_experts, 
        parts=ep_size, 
        index=coord.ep, 
        require_even=True,
    )


def tensor_parallel_shard(shape: tuple[int,...], dim: int, coord:RankCoord, tp_size: int, *, require_even: bool = True,) -> TensorShard:
    
    if len(shape) == 0:
        raise ValueError("shape must have at least one dimension")
   
    if dim < 0:  
        dim = len(shape) + dim 

    if dim < 0 or dim >= len(shape):  
        raise ValueError(f"dim must be in [0, {len(shape)}), got {dim}")  
    
    global_size = shape[dim]
    
    index_range = split_range(
        size=global_size, 
        parts=tp_size, 
        index=coord.tp, 
        require_even=require_even,
    )

    return TensorShard(
        dim=dim, 
        index_range=index_range, 
        global_size=global_size, 
        num_shards=tp_size, 
        shard_index=coord.tp,  
    ) 


def placement_for_rank( rank: int, mesh: RankMesh, *, num_layers: int, global_batch_size: int, sequence_length: int, tensor_shape: tuple[int, ...] | None = None, tensor_dim: int = -1, num_experts: int | None = None,) -> RankPlacement:
    """
    RankPlacement is the readable ownership summary for one flat rank:
    rank -> mesh coordinate -> layer/batch/context/expert/tensor ownership.
    """
    coord = mesh.coord_for_rank(rank)

    return RankPlacement(
        rank=rank,
        coord=coord,
        layers=pipeline_layers(num_layers, coord, mesh.dims.pp),
        batch=data_batch_range(global_batch_size, coord, mesh.dims.dp),
        context=context_range(sequence_length, coord, mesh.dims.cp),
        experts=(
            expert_range(num_experts, coord, mesh.dims.ep)
            if num_experts is not None
            else None
        ),
        tensor=(
            tensor_parallel_shard(tensor_shape, tensor_dim, coord, mesh.dims.tp)
            if tensor_shape is not None
            else None
        ),
    )


if __name__ == "__main__":  
    runtime = init_runtime()

    try:
        runtime_dims = ParallelDims()
        runtime_mesh = RankMesh(runtime_dims, world_size=runtime.world_size)
        runtime_groups = build_parallel_groups(runtime, runtime_mesh)
        runtime_placement = placement_for_rank(
            runtime.rank,
            runtime_mesh,
            num_layers=24,
            global_batch_size=1024,
            sequence_length=8192,
            tensor_shape=(4096, 16384),
            tensor_dim=1,
        )

        print("runtime:", runtime)
        print("runtime groups:", runtime_groups)
        print("runtime placement:", runtime_placement)

        demo_dims = ParallelDims(dp=2, pp=2, tp=2)
        demo_mesh = RankMesh(demo_dims)
        demo_placement = placement_for_rank(
            5,
            demo_mesh,
            num_layers=24,
            global_batch_size=1024,
            sequence_length=8192,
            tensor_shape=(4096, 16384),
            tensor_dim=1,
        )

        print("demo placement:", demo_placement)
    finally:
        shutdown_runtime()
