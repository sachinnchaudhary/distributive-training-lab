from __future__ import annotations 
from dataclasses import dataclass  
import socket
import os
import torch    
import torch.distributed as dist  


@dataclass(frozen=True)  
class RuntimeContext:  

    rank: int  
    local_rank: int  
    world_size: int  
    local_world_size: int  
    node_rank: int  
    device: torch.device  
    backend: str  
    hostname: str  
    is_distributed: bool  


_RUNTIME: RuntimeContext | None = None  

def get_runtime() -> RuntimeContext: 

    if _RUNTIME is None: 
        raise RuntimeError("runtime has not been initialized") 
    return _RUNTIME  


def _read_int_env(name, default=None):  
    value = os.environ.get(name) 

    if value is None: 
        return default  
    try:
        return int(value)  
    except ValueError: 
        raise ValueError(f"{name} must be an integer, got {value!r}") from None  



def _discover_rank_info():  

   rank =  _read_int_env("RANK", 0)
   local_rank = _read_int_env("LOCAL_RANK", 0)  
   world_size = _read_int_env("WORLD_SIZE", 1)  

   local_world_size = _read_int_env("LOCAL_WORLD_SIZE", None)  
   if local_world_size is None:  
       if torch.cuda.is_available():  
           local_world_size = torch.cuda.device_count()  
       else: 
           local_world_size =1  

   if rank < 0:
       raise ValueError(f"rank must be non-negative, got {rank}")
   if local_rank < 0:
       raise ValueError(f"local_rank must be non-negative, got {local_rank}")
   if world_size < 1:
       raise ValueError(f"world_size must be at least 1, got {world_size}")
   if local_world_size < 1:
       raise ValueError(f"local_world_size must be at least 1, got {local_world_size}")

   node_rank = _read_int_env("NODE_RANK", None) 
   if node_rank is None:  
       node_rank = _infer_node_rank(rank, local_world_size)  
   elif node_rank < 0:
       raise ValueError(f"node_rank must be non-negative, got {node_rank}")

   return rank, local_rank, world_size, local_world_size, node_rank



def _infer_node_rank(rank, local_world_size): 
    return rank // local_world_size  


def _select_device(local_rank): 
     if torch.cuda.is_available():  
         device_count = torch.cuda.device_count()  

         if local_rank >= device_count:  
             raise ValueError(
                 f"local_rank {local_rank} is not valid for {device_count} CUDA devices"
             )  

         torch.cuda.set_device(local_rank) 
         return torch.device("cuda", local_rank)  

     return torch.device("cpu")  


def _select_backend(device, backend= None):  
    if backend is not None: 
        return backend 
    if device.type == "cuda":  
        return "nccl"  
    
    return "gloo"


def _init_process_group_if_needed(backend, rank, world_size):  

    if world_size == 1:  
        return False  
    if dist.is_initialized(): 
        return True 

    dist.init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size
    )      

    return True 


def init_runtime(backend=None):  

    global _RUNTIME  

    if _RUNTIME is not None:  
        return _RUNTIME  
    
    rank, local_rank, world_size, local_world_size, node_rank = _discover_rank_info()

    device = _select_device(local_rank) 
    backend = _select_backend(device, backend)
    hostname = socket.gethostname() 

    is_distributed = _init_process_group_if_needed(
        backend=backend, 
        rank=rank, 
        world_size=world_size
    )
    
    _RUNTIME = RuntimeContext(
        rank=rank, 
        local_rank=local_rank, 
        world_size= world_size, 
        local_world_size=local_world_size, 
        node_rank=node_rank, 
        device=device, 
        backend=backend, 
        hostname=hostname, 
        is_distributed=is_distributed, 
    )
    
    return _RUNTIME  


def shutdown_runtime(): 

    global _RUNTIME 

    if dist.is_available() and dist.is_initialized():  
          dist.destroy_process_group() 
    
    _RUNTIME = None  


