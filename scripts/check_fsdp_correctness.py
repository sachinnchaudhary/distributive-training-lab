from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meshtrain.core.distributed.groups import build_parallel_groups
from meshtrain.core.distributed.mesh import ParallelDims, RankMesh
from meshtrain.core.distributed.runtime import init_runtime, shutdown_runtime
from meshtrain.parallelism.data_parallel import FSDPConfig, FullyShardedModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--reshard-after-forward", action="store_true", default=True)
    return parser.parse_args()


def broadcast_module_parameters(module: nn.Module, groups) -> None:
    if groups.dp_group is None or len(groups.dp_ranks) <= 1:
        return

    src_rank = groups.dp_ranks[0]
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src_rank, group=groups.dp_group)


def max_output_diff(local_output: torch.Tensor, reference_output: torch.Tensor) -> float:
    return float((local_output.detach() - reference_output.detach()).abs().max().item())


def main() -> None:
    args = parse_args()
    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=runtime.world_size, pp=1, tp=1, cp=1, ep=1)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        torch.manual_seed(args.seed)
        reference = nn.Linear(
            args.in_features,
            args.out_features,
            bias=False,
        ).to(runtime.device)
        broadcast_module_parameters(reference, groups)

        wrapped_module = copy.deepcopy(reference)
        fsdp_module = FullyShardedModule(
            wrapped_module,
            groups,
            FSDPConfig(reshard_after_forward=args.reshard_after_forward),
        ).to(runtime.device)

        torch.manual_seed(args.seed + 1)
        x = torch.randn(
            args.batch_size,
            args.in_features,
            device=runtime.device,
        )

        reference_output = reference(x)
        fsdp_output = fsdp_module(x)

        diff = torch.tensor(
            max_output_diff(fsdp_output, reference_output),
            device=runtime.device,
        )
        if runtime.is_distributed:
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)

        passed = bool(diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  input_shape={tuple(x.shape)}")
                print(f"  output_shape={tuple(fsdp_output.shape)}")
                print(f"  local_param_shard_numel={fsdp_module.local_param_shard.numel()}")
                print(f"  total_module_numel={fsdp_module.total_numel}")
                print(f"  max_output_diff={float(diff.item())}")
            if runtime.is_distributed:
                dist.barrier()

        if runtime.rank == 0:
            print(f"passed: {global_passed}")
        if not global_passed:
            raise SystemExit(1)

    finally:
        shutdown_runtime()


if __name__ == "__main__":
    main()
