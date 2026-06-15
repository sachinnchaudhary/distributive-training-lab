from __future__ import annotations 
from dataclasses import dataclass  

import torch  
import torch.distributed as dist  

from .groups import ParallelGroups
from .mesh import RankCoord  
from .collective_trace import (
    CollectiveTraceConfig,
    CollectiveTraceEvent,
    CollectiveTracer,
    TensorTrace,
)

@dataclass(frozen=True)
class CollectiveDebug:  
    enabled: bool = False  
    rank: int | None = None  
    coord: RankCoord | None = None 
    label: str | None = None  


def _group_is_active(group: dist.ProcessGroup | None, ranks: list[int]):  
    return group is not None and len(ranks) > 1  

def _prepare_output(tensor: torch.Tensor, inplace: bool):  

    if inplace: 
        return tensor  
    
    return tensor.clone()  


def _normalize_dim(dim: int, ndim: int) -> int:  
    if ndim == 0:  
        raise ValueError("cannot use dim on a scalar tensor")  
    
    if dim < 0:  
        dim = ndim + dim  

    if dim < 0 or dim >= ndim:  
        raise ValueError(f"dim must be in [0, {ndim}), got {dim}") 
    
    return dim  


def _debug_prefix(debug: CollectiveDebug) -> str:   

    parts = []  

    if debug.rank is not None:  
        parts.append(f"rank {debug.rank}") 
    
    if debug.coord is not None:  
        parts.append(
            f"dp={debug.coord.dp} pp={debug.coord.pp} tp={debug.coord.tp} "
            f"cp={debug.coord.cp} ep={debug.coord.ep}"
        )
    
    return "[" + " | ".join(parts) + "]"  


def _log_collective(debug: CollectiveDebug | None, 
*, 
operation: str, 
group_name: str, 
group_ranks: list[int], 
input_shape: tuple[int, ...], 
output_shape: tuple[int,...] | None = None, 
inplace: bool | None=None, 
status: str = "run", 
) -> None: 

    if debug is None or not debug.enabled:  
        return  

    prefix = _debug_prefix(debug)  
    label =  f"\n{debug.label}" if debug.label else ""  
    
    print(prefix)  
    if label:
        print(label)
    print(operation)
    print(f"group = {group_name}_group {group_ranks}")
    print(f"input_shape = {list(input_shape)}")
    if output_shape is not None:
        print(f"output_shape = {list(output_shape)}")
    if inplace is not None:
        print(f"inplace = {inplace}")
    print(f"status = {status}")


def _record_trace(
        tracer: CollectiveTracer | None,  
        trace_event: CollectiveTraceEvent | None,         
) -> None:  
    if tracer is None or trace_event is None:  
        return   
    
    tracer.record(trace_event)  

def _all_reduce(
    tensor: torch.Tensor,  
    *, 
    group: dist.ProcessGroup | None, 
    ranks: list[int], 
    group_name: str, 
    operation: str, 
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    inplace: bool = True, 
    debug: CollectiveDebug | None = None,  
    tracer: CollectiveTracer | None = None,  
    trace_event: CollectiveTraceEvent | None = None
) -> torch.Tensor:  
    
    output = _prepare_output(tensor, inplace) 

    if not _group_is_active(group, ranks):  
        _log_collective( 
            debug,
            operation=operation,
            group_name=group_name, 
            group_ranks=ranks, 
            input_shape=tuple(tensor.shape), 
            output_shape=tuple(output.shape), 
            inplace=inplace, 
            status="no-op", 
        )
        _record_trace(tracer, trace_event)
        
        return output 

    dist.all_reduce(
        output, 
        op=op, 
        group=group, 
    )  

    _log_collective(
        debug,
        operation=operation, 
        group_name=group_name, 
        group_ranks=ranks, 
        input_shape=tuple(tensor.shape), 
        output_shape=tuple(output.shape), 
        inplace=inplace, 
        status="run",
    ) 
    _record_trace(tracer, trace_event) 
    return output 


def all_reduce_tensor_parallel(
        tensor: torch.Tensor, 
        groups: ParallelGroups, 
        *, 
        op: dist.ReduceOp = dist.ReduceOp.SUM, 
        inplace: bool = True, 
        debug: CollectiveDebug | None = None, 
        tracer: CollectiveTracer | None = None,  
        trace_event: CollectiveTraceEvent | None = None 
) -> torch.Tensor:  
    
    return _all_reduce(
        tensor, 
        group=groups.tp_group, 
        ranks= groups.tp_ranks, 
        group_name="tp", 
        operation = "all_reduce_tensor_parallel", 
        op=op, 
        inplace=inplace, 
        debug=debug,
        tracer = tracer, 
        trace_event= trace_event, 
    ) 


def all_reduce_data_parallel(
        tensor: torch.Tensor,
        groups: ParallelGroups, 
        *, 
        op: dist.ReduceOp = dist.ReduceOp.SUM, 
        average: bool = False, 
        inplace: bool = True, 
        debug: CollectiveDebug | None = None, 
        tracer: CollectiveTracer | None = None,  
        trace_event: CollectiveTraceEvent | None = None
) -> torch.Tensor:
    
    output = _all_reduce( 
        tensor, 
        group= groups.dp_group, 
        ranks=groups.dp_ranks, 
        group_name="dp", 
        operation="all_reduce_data_parallel", 
        op=op, 
        inplace=inplace, 
        debug=debug,
        tracer = tracer, 
        trace_event= trace_event,
    )

    if average and len(groups.dp_ranks) > 1:  
        output.div_(len(groups.dp_ranks))  
    
    return output  


def all_gather_tensor_parallel(
        tensor: torch.Tensor, 
        groups: ParallelGroups, 
        *, 
        dim: int = -1, 
        debug: CollectiveDebug | None = None,
        tracer: CollectiveTracer | None = None,  
        trace_event: CollectiveTraceEvent | None = None  
) -> torch.Tensor:  
    
     dim = _normalize_dim(dim, tensor.ndim) 

     if not _group_is_active(groups.tp_group, groups.tp_ranks):  
         _log_collective(
             debug, 
             operation="all_gather_tensor_parallel", 
             group_name="tp", 
             group_ranks=groups.tp_ranks, 
             input_shape=tuple(tensor.shape), 
             output_shape=tuple(tensor.shape),  
             inplace=False, 
             status="no-op",
         )

         _record_trace(tracer, trace_event)
         return tensor
     
     gathered = [torch.empty_like(tensor) for _ in groups.tp_ranks]

     dist.all_gather(
         gathered, 
         tensor, 
         group=groups.tp_group
     )
     
     output = torch.cat(gathered, dim=dim)  

     _log_collective(
         debug, 
         operation="all_gather_tensor_parallel", 
         group_name="tp", 
         group_ranks=groups.tp_ranks, 
         input_shape=tuple(tensor.shape), 
         output_shape=tuple(output.shape), 
         inplace=False, 
         status="run",
         )
     _record_trace(tracer, trace_event)  
     return output   


def reduce_scatter_tensor_parallel(
        tensor: torch.Tensor,  
        groups:  ParallelGroups, 
        *, 
        dim: int = -1, 
        op: dist.ReduceOp = dist.ReduceOp.SUM, 
        debug: CollectiveDebug | None = None,  
        tracer: CollectiveTracer | None = None,
        trace_event: CollectiveTraceEvent | None = None,
) -> torch.Tensor:  

     dim = _normalize_dim(dim, tensor.ndim) 

     if not _group_is_active(groups.tp_group, groups.tp_ranks):  
        
          _log_collective(
             debug, 
             operation="reduce_scatter_tensor_parallel", 
             group_name="tp", 
             group_ranks=groups.tp_ranks, 
             input_shape=tuple(tensor.shape), 
             output_shape=tuple(tensor.shape),  
             inplace=False, 
             status="no-op",
         )

          _record_trace(tracer, trace_event)
          return tensor
     
     world = len(groups.tp_ranks)  

     if tensor.shape[dim] % world != 0:  
         raise ValueError("reduce_scatter dim must divide by tp world size")  

     chunks = list(torch.chunk(tensor, world, dim=dim))  
     output = torch.empty_like(chunks[0])   

     dist.reduce_scatter(
         output, 
         chunks, 
         op=op,  
         group=groups.tp_group,
     )

     _log_collective(
         debug, 
         operation="reduce_scatter_tensor_parallel", 
         group_name="tp", 
         group_ranks=groups.tp_ranks, 
         input_shape=tuple(tensor.shape), 
         output_shape=tuple(output.shape), 
         inplace=False, 
         status="run",
         )
     
     _record_trace(tracer, trace_event)
     return output


def all_to_all_expert_tokens(
        tensor: torch.Tensor,
        groups: ParallelGroups,
        *,
        dim: int = 0,
        debug: CollectiveDebug | None = None,
        tracer: CollectiveTracer | None = None,
        trace_event: CollectiveTraceEvent | None = None,
) -> torch.Tensor:

     dim = _normalize_dim(dim, tensor.ndim)

     if not _group_is_active(groups.ep_group, groups.ep_ranks):
         """
         NOTE: ep=1 means all experts are local to this rank, so token routing
         across expert-parallel ranks is a no-op.
         """
         _log_collective(
             debug,
             operation="all_to_all_expert_tokens",
             group_name="ep",
             group_ranks=groups.ep_ranks,
             input_shape=tuple(tensor.shape),
             output_shape=tuple(tensor.shape),
             inplace=False,
             status="no-op",
         )
         _record_trace(tracer, trace_event)
         return tensor

     world = len(groups.ep_ranks)

     if tensor.shape[dim] % world != 0:
         raise ValueError("all_to_all dim must divide by ep world size")

     input_chunks = list(torch.chunk(tensor, world, dim=dim))
     output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world)]

     dist.all_to_all(
         output_chunks,
         input_chunks,
         group=groups.ep_group,
     )

     output = torch.cat(output_chunks, dim=dim)

     _log_collective(
         debug,
         operation="all_to_all_expert_tokens",
         group_name="ep",
         group_ranks=groups.ep_ranks,
         input_shape=tuple(tensor.shape),
         output_shape=tuple(output.shape),
         inplace=False,
         status="run",
     )

     _record_trace(tracer, trace_event)
     return output


def send_pipeline_activation(
        tensor: torch.Tensor, 
        dst_rank: int, 
        *, 
        debug: CollectiveDebug | None = None,  
        tracer: CollectiveTracer | None = None,
        trace_event: CollectiveTraceEvent | None = None,
) -> None:  

     dist.send(tensor, dst=dst_rank) 

     _log_collective(
         debug, 
         operation="send_pipeline_activation", 
         group_name="pp", 
         group_ranks=[dst_rank], 
         input_shape= tensor.shape, 
         output_shape= None, 
         status="run",
         )
     _record_trace(tracer, trace_event)
     

def recv_pipeline_activation( 
        src_rank: int,  
        *,  
        shape: tuple[int, ...],  
        dtype: torch.dtype,  
        device: torch.device | str,    
        debug: CollectiveDebug | None = None,  
        tracer: CollectiveTracer | None = None,
        trace_event: CollectiveTraceEvent | None = None,
)  -> torch.Tensor:   
    

    output = torch.empty(shape, dtype=dtype, device=device)  
    dist.recv(output, src=src_rank)  

    _log_collective(
         debug, 
         operation="recv_pipeline_activation", 
         group_name="pp", 
         group_ranks=[src_rank], 
         input_shape= shape, 
         output_shape= output.shape, 
         status="run",
         )  
    
    _record_trace(tracer, trace_event)
    return output  


if __name__ == "__main__":
    groups = ParallelGroups(
        rank=0,
        dp_ranks=[0],
        pp_ranks=[0],
        cp_ranks=[0],
        ep_ranks=[0],
        tp_ranks=[0],
        dp_group=None,
        pp_group=None,
        cp_group=None,
        ep_group=None,
        tp_group=None,
    )

    debug = CollectiveDebug(
        enabled=False,
        rank=0,
        coord=RankCoord(dp=0, pp=0, tp=0, cp=0, ep=0),
        label="collectives no-op sanity",
    )

    x = torch.ones(2, 4)
    tracer = CollectiveTracer(
        CollectiveTraceConfig(
            enabled=True,
            jsonl_path="meshtrain/core/logging/collectives_main_test.jsonl",
            rank_filter={0},
            axis_filter={"tp"},
            op_filter={"all_reduce"},
        )
    )
    trace_event = CollectiveTraceEvent(
        step=0,
        microbatch=0,
        rank=0,
        coord=debug.coord,
        layer=3,
        module="mlp.down_proj",
        op="all_reduce",
        axis="tp",
        group=groups.tp_ranks,
        reason="merge_row_parallel_output",
        before=TensorTrace(
            name="mlp_down_partial_output",
            meaning="partial_hidden_sum",
            local_shape=tuple(x.shape),
            global_shape=tuple(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            complete=False,
        ),
        after=TensorTrace(
            name="mlp_down_output",
            meaning="complete_hidden_state",
            local_shape=tuple(x.shape),
            global_shape=tuple(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            complete=True,
        ),
    )

    tp_reduced = all_reduce_tensor_parallel(
        x,
        groups,
        debug=debug,
        tracer=tracer,
        trace_event=trace_event,
    )
    dp_reduced = all_reduce_data_parallel(x, groups, average=True, debug=debug)
    gathered = all_gather_tensor_parallel(x, groups, dim=-1, debug=debug)
    scattered = reduce_scatter_tensor_parallel(x, groups, dim=-1, debug=debug)
    routed = all_to_all_expert_tokens(x, groups, dim=0, debug=debug)

    print("tp_reduced same object:", tp_reduced is x)
    print("dp_reduced same object:", dp_reduced is x)
    print("gathered shape:", tuple(gathered.shape))
    print("scattered shape:", tuple(scattered.shape))
    print("routed shape:", tuple(routed.shape))
    print("trace path:", tracer.config.jsonl_path)
    print("trace events written:", tracer.events_written)






  
