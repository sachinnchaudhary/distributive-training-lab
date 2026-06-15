from __future__ import annotations  
from dataclasses import dataclass
  

AXES = ("dp", "pp", "cp", "ep", "tp")  


@dataclass(frozen=True)
class ParallelDims:  

    dp: int = 1 
    pp: int = 1 
    tp: int = 1 
    cp: int = 1 
    ep: int = 1  

    def __post_init__(self):  

         for axis in AXES:  
           value = getattr(self, axis)  
           if value < 1:  
               raise ValueError(f"{axis} must be >=1")  
         
    @property
    def world_size(self): 
        return self.dp * self.pp * self.cp * self.ep * self.tp    


    @property 
    def shape(self):  
    
      return tuple(getattr(self, axis) for axis in AXES) 

    @property
    def axes(self):  
        return AXES  


@dataclass(frozen=True)  
class RankCoord:  
    dp: int 
    pp: int 
    tp: int 
    cp: int 
    ep: int 

    def get(self, axis):  
      
        if axis not in AXES:
            raise ValueError(f"unknown parallel axis {axis!r}; expected one of {AXES}")
        return getattr(self, axis)  
    
    def replace(self, axis, value):  
       
        if axis not in AXES:
            raise ValueError(f"unknown parallel axis {axis!r}; expected one of {AXES}")
        return RankCoord(
            dp=value if axis == "dp" else self.dp, 
            pp=value if axis == "pp" else self.pp, 
            tp = value if axis=="tp" else self.tp, 
            cp=value if axis == "cp" else self.cp, 
            ep=value if axis=="ep" else self.ep
        )



class RankMesh:   
    def __init__(self, dims: ParallelDims, world_size: int | None = None):  
        expected_world_size = dims.world_size  

        if world_size is not None and world_size != expected_world_size:  
            raise ValueError(
                f"parallel dims require world_size {expected_world_size}, got {world_size}" 
            )
        
        self.dims = dims 
        self.world_size = expected_world_size
        self.axes = AXES  


    def _validate_axis(self, axis: str) -> None:
      
         if axis not in self.axes:
             raise ValueError(f"unknown parallel axis {axis!r}; expected one of {self.axes}")


    def _validate_rank(self, rank: int) -> None: 
         if rank < 0 or rank >= self.world_size:  
             raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}") 


    def _validate_coord(self, coord: RankCoord) -> None:
         for axis in self.axes:
             value = coord.get(axis)
             axis_size = getattr(self.dims, axis)
             if value < 0 or value >= axis_size:
                 raise ValueError(
                     f"{axis} coordinate must be in [0, {axis_size}), got {value}"
                 )
        
    
    def coord_for_rank(self, rank: int) -> RankCoord: 
        self._validate_rank(rank)  

        remaining = rank  
        values = {}  

        for axis in reversed(self.axes):  
            axis_size = getattr(self.dims, axis) 
            values[axis]  = remaining % axis_size  
            remaining = remaining // axis_size  

        return RankCoord(
            dp=values["dp"], 
            pp=values["pp"],  
            tp=values["tp"], 
            cp=values["cp"], 
            ep=values["ep"],
         )      


    def rank_for_coord(self, coord: RankCoord) -> int:
        self._validate_coord(coord)

        rank = 0
        for axis in self.axes:
            axis_size = getattr(self.dims, axis)
            rank = rank * axis_size + coord.get(axis)

        return rank
     
    def ranks_along_axis(self, rank: int, axis: str) -> list[int]:  
        self._validate_rank(rank) 
        self._validate_axis(axis)  

        base_coord = self.coord_for_rank(rank)  
        axis_size = getattr(self.dims, axis)  

        peers = []  

        for axis_index in range(axis_size):  
            peer_coord = base_coord.replace(axis, axis_index) 
            peer_rank = self.rank_for_coord(peer_coord)  
            peers.append(peer_rank)  

        return peers

    def groups(self, axis: str) -> list[list[int]]:
      """
      NOTE: groups.py needs every axis group in deterministic order so every
      rank calls torch.distributed.new_group in the same sequence.
      """

      self._validate_axis(axis)    
      
      seen = set()  
      all_groups = []  

      for rank in range(self.world_size):  
          group = tuple(self.ranks_along_axis(rank, axis))  
      
          if group in seen:  
              continue  
          
          seen.add(group)
          all_groups.append(list(group))
          
      return all_groups

    def group(self, axis: str) -> list[list[int]]:
      """
      NOTE: temporary compatibility alias for older local calls; the public
      name is groups(axis) because it returns all groups along that axis.
      """
      return self.groups(axis)

          
if __name__ == "__main__":  
    dims = ParallelDims(dp=1, pp=4, tp=2)  
    mesh = RankMesh(dims) 
    
    coord = mesh.coord_for_rank(5)
    peers = mesh.ranks_along_axis(5, axis="tp") 
    groups = mesh.group("tp")
    print(coord)  
    print(peers)
    print(groups)  



