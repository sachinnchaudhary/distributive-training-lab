from __future__ import annotations  

import argparse   
import sys
from pathlib import Path
from types import SimpleNamespace 

import torch.distributed as dist  

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meshtrain.core.distributed.runtime import init_runtime, shutdown_runtime
from meshtrain.core.distributed.mesh import ParallelDims, RankMesh  
from meshtrain.core.distributed.groups import build_parallel_groups  
from meshtrain.core.distributed.placement import placement_for_rank  


def parse_args():  

    parser = argparse.ArgumentParser()  
    
    #mode
    parser.add_argument("--mode", choices=["real", "demo"], default="demo")  
    
    #mesh dims
    parser.add_argument("--dp", type=int, default=2) 
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1) 

    #demo only  
    parser.add_argument("--demo-rank", type=int, default=5)

    # placement config
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--tensor-shape", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--tensor-dim", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=None)

    return parser.parse_args()  


def format_range(index_range):  

    return f"[{index_range.start}, {index_range.end})"


def print_rank_report(
      *,
    rank,
    physical,
    coord,
    groups,
    placement,     
): 

    print(f"rank={rank}")
    print()

    print("physical:")
    print(f"  node={physical.node_rank}")
    print(f"  local_rank={physical.local_rank}")
    print(f"  device={physical.device}")
    print()

    print("logical:")
    print(f"  dp={coord.dp}")
    print(f"  pp={coord.pp}")
    print(f"  tp={coord.tp}")
    print(f"  cp={coord.cp}")
    print(f"  ep={coord.ep}")
    print()

    print("groups:")
    print(f"  dp_group={groups.dp_ranks}")
    print(f"  pp_group={groups.pp_ranks}")
    print(f"  tp_group={groups.tp_ranks}")
    print(f"  cp_group={groups.cp_ranks}")
    print(f"  ep_group={groups.ep_ranks}")
    print()

    print("ownership:")
    print(f"  data_shard={coord.dp}")
    print(f"  layer_stage={coord.pp}")
    print(f"  tensor_shard={coord.tp}")
    print(f"  layers={format_range(placement.layers.layers)}")
    print(f"  batch={format_range(placement.batch)}")
    print(f"  context={format_range(placement.context)}")

    if placement.experts is not None:
        print(f"  experts={format_range(placement.experts)}")

    if placement.tensor is not None:
        dim = placement.tensor.dim
        print(f"  tensor_dim_{dim}={format_range(placement.tensor.index_range)}")  



def run_real(args):  

    runtime = init_runtime()  
    try:  
        dims = ParallelDims( 
            dp=args.dp,  
            pp=args.pp, 
            tp=args.tp, 
            cp=args.cp,  
            ep=args.ep,  
        )  
    
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)  
        coord = mesh.coord_for_rank(runtime.rank)  

        placement = placement_for_rank(   
            runtime.rank,
            mesh,
            num_layers=args.num_layers,
            global_batch_size=args.global_batch_size,
            sequence_length=args.sequence_length,
            tensor_shape=tuple(args.tensor_shape),
            tensor_dim=args.tensor_dim,
            num_experts=args.num_experts,
        )
        
        for r in range(runtime.world_size):  
            if runtime.rank == r:  
                print_rank_report(
                    rank=runtime.rank,
                    physical=runtime,
                    coord=coord,
                    groups=groups,
                    placement=placement,
                )
            if runtime.is_distributed:  
                dist.barrier()  
    finally: 
        shutdown_runtime()  


def run_demo(args):  

    dims = ParallelDims(
        dp= args.dp, 
        pp=args.pp,  
        tp=args.tp,  
        cp=args.cp, 
        ep=args.ep, 
    )

    mesh = RankMesh(dims) 
    rank = args.demo_rank  
    coord = mesh.coord_for_rank(rank)  

    placement = placement_for_rank(
        rank, 
        mesh,  
        num_layers=args.num_layers, 
        global_batch_size=args.global_batch_size, 
        sequence_length=args.sequence_length, 
        tensor_shape=tuple(args.tensor_shape), 
        tensor_dim=args.tensor_dim, 
        num_experts=args.num_experts, 
    ) 

    physical = SimpleNamespace(
        node_rank=0,  
        local_rank=rank,  
        device = f"demo:rank{rank}"
    )

    groups =  SimpleNamespace(  
      dp_ranks = mesh.ranks_along_axis(rank, "dp"),  
      pp_ranks = mesh.ranks_along_axis(rank, "pp"),
      tp_ranks = mesh.ranks_along_axis(rank, "tp"),
      cp_ranks = mesh.ranks_along_axis(rank, "cp"),
      ep_ranks = mesh.ranks_along_axis(rank, "ep"),
      )

    print_rank_report(
        rank=rank,
        physical=physical,
        coord=coord,
        groups=groups,
        placement=placement,
    )


def main():  

    args = parse_args()  

    if args.mode == "demo":  
        run_demo(args)  

    else:  
        run_real(args)  


if __name__ == "__main__":  

    main()

