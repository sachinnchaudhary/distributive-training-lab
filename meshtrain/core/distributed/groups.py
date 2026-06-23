from __future__ import annotations  
from dataclasses import dataclass  
import torch.distributed as dist  
from .runtime import RuntimeContext  
from .mesh import RankCoord, RankMesh 


GROUP_AXIS_ORDER = ("dp", "pp", "cp", "ep", "tp")  


@dataclass(frozen=True)  
class ParallelGroups:  

    rank: int 

    dp_ranks: list[int] 
    pp_ranks: list[int]  
    cp_ranks: list[int]  
    ep_ranks: list[int]  
    tp_ranks: list[int]  
    stage_ranks: list[int]

    dp_group: dist.ProcessGroup | None  
    pp_group: dist.ProcessGroup | None  
    cp_group: dist.ProcessGroup | None 
    ep_group: dist.ProcessGroup | None  
    tp_group: dist.ProcessGroup | None  
    stage_group: dist.ProcessGroup | None

    

def _single_process_groups(rank: int) -> ParallelGroups:    
         return ParallelGroups( 
               rank=rank,  

               dp_ranks=[rank], 
               pp_ranks =[rank], 
               cp_ranks=[rank], 
               ep_ranks = [rank], 
               tp_ranks=[rank],  
               stage_ranks=[rank],

               dp_group=None, 
               pp_group=None, 
               cp_group=None, 
               ep_group=None,  
               tp_group=None,   
               stage_group=None,
         )


def _stage_groups(mesh: RankMesh) -> list[list[int]]:
     groups: list[list[int]] = []

     for dp_index in range(mesh.dims.dp):
          for pp_index in range(mesh.dims.pp):
               ranks: list[int] = []

               for cp_index in range(mesh.dims.cp):
                    for ep_index in range(mesh.dims.ep):
                         for tp_index in range(mesh.dims.tp):
                              ranks.append(
                                   mesh.rank_for_coord(
                                        RankCoord(
                                             dp=dp_index,
                                             pp=pp_index,
                                             cp=cp_index,
                                             ep=ep_index,
                                             tp=tp_index,
                                        )
                                   )
                              )

               groups.append(ranks)

     return groups
    

def build_parallel_groups(runtime: RuntimeContext, mesh: RankMesh) -> ParallelGroups: 
         rank = runtime.rank 

         if mesh.world_size != runtime.world_size:
              raise ValueError(
                   f"mesh world_size {mesh.world_size} does not match runtime world_size {runtime.world_size}"
              )

         if not runtime.is_distributed:  
              return _single_process_groups(rank)

         if not dist.is_initialized():
              raise RuntimeError(
                   "torch.distributed must be initialized before building parallel groups"
              )
         
         created = {}  
         
         for axis in GROUP_AXIS_ORDER:   
              created[axis] = {  
                   "ranks": None,  
                   "group": None,  
              } 
                    
              axis_size = getattr(mesh.dims, axis)

              if axis_size == 1:
                  
                   created[axis]["ranks"] = [rank]
                   created[axis]["group"] = None
                   continue

              all_axis_groups = mesh.groups(axis)

              for ranks in all_axis_groups: 
                   group = dist.new_group(ranks=ranks) 
                  
                   if rank in ranks:  
                         created[axis]["ranks"] = ranks  
                         created[axis]["group"] = group  

         for axis in GROUP_AXIS_ORDER:
              if created[axis]["ranks"] is None:
                   raise RuntimeError(f"rank {rank} was not assigned to a {axis} group")

         stage_ranks = [rank]
         stage_group = None
         for ranks in _stage_groups(mesh):
              group = None if len(ranks) == 1 else dist.new_group(ranks=ranks)
              if rank in ranks:
                   stage_ranks = ranks
                   stage_group = group

         return ParallelGroups(
              rank=rank,  

              dp_ranks=created["dp"]["ranks"], 
              dp_group=created["dp"]["group"],  

              pp_ranks=created["pp"]["ranks"],  
              pp_group=created["pp"]["group"],

              cp_ranks=created["cp"]["ranks"],  
              cp_group=created["cp"]["group"],

              ep_ranks=created["ep"]["ranks"],  
              ep_group=created["ep"]["group"],

              tp_ranks=created["tp"]["ranks"],  
              tp_group=created["tp"]["group"],

              stage_ranks=stage_ranks,
              stage_group=stage_group,

         )      
               
         
