from __future__ import annotations

import argparse
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
from meshtrain.parallelism.expert_parallelism import (
    combine_expert_outputs,
    dispatch_tokens_to_experts,
    expert_parallel_range,
    run_local_experts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def build_experts(num_experts: int, hidden_dim: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            )
            for _ in range(num_experts)
        ]
    )


def broadcast_module_parameters(module: nn.Module, group, src_rank: int) -> None:
    if group is None:
        return

    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src_rank, group=group)


def reference_expert_output(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    output = torch.empty_like(tokens)

    for expert_id, expert in enumerate(experts):
        mask = expert_ids == expert_id
        if not bool(mask.any()):
            continue
        output[mask] = expert(tokens[mask])

    return output


def main() -> None:
    args = parse_args()
    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=1, pp=1, tp=1, cp=1, ep=runtime.world_size)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        torch.manual_seed(args.seed)
        experts = build_experts(args.num_experts, args.hidden_dim).to(runtime.device)

        if runtime.is_distributed:
            broadcast_module_parameters(experts, groups.ep_group, groups.ep_ranks[0])

        torch.manual_seed(args.seed + 1)
        tokens = torch.randn(
            args.num_tokens,
            args.hidden_dim,
            device=runtime.device,
        )
        expert_ids = torch.arange(
            args.num_tokens,
            device=runtime.device,
            dtype=torch.long,
        ) % args.num_experts
        expert_ids = expert_ids.roll(shifts=runtime.rank)

        expected = reference_expert_output(tokens, expert_ids, experts)

        shard = expert_parallel_range(args.num_experts, groups)
        local_experts = nn.ModuleList(
            [
                experts[expert_id]
                for expert_id in range(shard.start, shard.end)
            ]
        )

        dispatch = dispatch_tokens_to_experts(
            tokens,
            expert_ids,
            num_experts=args.num_experts,
            groups=groups,
        )
        local_outputs = run_local_experts(
            dispatch.received_tokens,
            dispatch.received_expert_ids,
            local_experts,
            shard,
        )
        actual = combine_expert_outputs(
            local_outputs,
            dispatch,
            original_num_tokens=args.num_tokens,
            groups=groups,
        )

        local_diff = (actual - expected).abs().max()
        global_diff = local_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(global_diff, op=dist.ReduceOp.MAX, group=groups.ep_group)

        passed = bool(global_diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN, group=groups.ep_group)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  ep_ranks={groups.ep_ranks}")
                print(f"  ep_rank={shard.ep_rank}")
                print(f"  expert_shard=[{shard.start}, {shard.end})")
                print(f"  expert_ids={expert_ids.tolist()}")
                print(f"  send_counts={dispatch.send_counts.tolist()}")
                print(f"  recv_counts={dispatch.recv_counts.tolist()}")
                print(f"  received_tokens={tuple(dispatch.received_tokens.shape)}")
                print(f"  local_max_diff={float(local_diff.item())}")
                print(f"  global_max_diff={float(global_diff.item())}")
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
