from __future__ import annotations  
from dataclasses import dataclass  
import json  
from pathlib import Path

from .mesh import RankCoord

DEFAULT_COLLECTIVE_TRACE_PATH = str(
    Path(__file__).resolve().parents[1] / "logging" / "collectives.jsonl"
)

@dataclass(frozen=True)  
class TensorTrace:  
    name: str
    meaning: str
    local_shape: tuple[int, ...]  
    global_shape: tuple[int, ...] | None = None  
    dtype: str | None = None  
    device: str | None = None  
    complete: bool = False  


@dataclass(frozen=True)  
class CollectiveTraceEvent:  
    step: int | None  
    microbatch: int | None  
    rank: int  
    coord: RankCoord | None 

    layer: int | None  
    module: str | None  

    op: str 
    axis: str 
    group: list[int]  
    reason: str  

    before: TensorTrace
    after: TensorTrace  


@dataclass(frozen=True)  
class CollectiveTraceConfig:  
    enabled: bool = False  
    jsonl_path: str | None = DEFAULT_COLLECTIVE_TRACE_PATH  

    rank_filter: set[int] | None = None  
    op_filter: set[str] | None = None  
    axis_filter: set[str] | None = None  
    module_filter: set[str] | None = None  

    every_n_steps: int = 1  
    max_events: int | None = None  


def _coord_to_dict(coord):  
    if coord is None:  
        return None  
    
    return {  
        "dp": coord.dp,  
        "pp": coord.pp,  
        "tp": coord.tp,  
        "cp": coord.cp,  
        "ep": coord.ep,  
    }


def _tensor_trace_to_dict(t): 
   
   return {  
       
       "name": t.name,  
       "meaning": t.meaning,  
       "local_shape": list(t.local_shape), 
       "global_shape": list(t.global_shape) if t.global_shape is not None else None,  
       "dtype": t.dtype,  
       "device": t.device,  
       "complete": t.complete,
   }


def event_to_dict(event):  
   
   return {
        "step": event.step,
        "microbatch": event.microbatch,
        "rank": event.rank,
        "coord": _coord_to_dict(event.coord),
        "layer": event.layer,
        "module": event.module,
        "op": event.op,
        "axis": event.axis,
        "group": event.group,
        "reason": event.reason,
        "before": _tensor_trace_to_dict(event.before),
        "after": _tensor_trace_to_dict(event.after),
    }

class CollectiveTracer:  

    def __init__(self, config: CollectiveTraceConfig):  
        self.config= config  
        self.events_written = 0   

        if config.enabled and config.jsonl_path is None:  
            raise ValueError("jsonl_path must be set when collective tracing is enabled")  
        
        if config.jsonl_path:  
           Path(config.jsonl_path).parent.mkdir(parents=True, exist_ok=True)            
     

    def should_record(self, event):  
       config = self.config  

       if not config.enabled:  
           return False  
       
       if config.max_events is not None and self.events_written >= config.max_events:  
           return False  
       
       if event.step is not None and event.step % config.every_n_steps != 0:  
           return False  
       
       if config.rank_filter is not None and event.rank not in config.rank_filter: 
           return False  
       
       if config.op_filter is not None and event.op not in config.op_filter: 
           return False  
       
       if config.axis_filter is not None and event.axis not in config.axis_filter:  
           return False  
       
       if config.module_filter is not None and event.module not in config.module_filter: 
           return False  
       

       return True 
    

    def record(self, event):  
        if not self.should_record(event):  
            return False  
        
        row = event_to_dict(event)  
        
        with open(self.config.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")  

        self.events_written += 1
        return True

