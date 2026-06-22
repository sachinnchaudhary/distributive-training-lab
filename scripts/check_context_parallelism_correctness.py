from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meshtrain.core.distributed.groups import build_parallel_groups
from meshtrain.core.distributed.mesh import ParallelDims, RankMesh
from meshtrain.core.distributed.runtime import init_runtime, shutdown_runtime
from meshtrain.parallelism.context_parallelism import (
    context_parallel_range,
    gather_sequence,
    ring_causal_attention,
    shard_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def full_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    seq_len = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
    )
    scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main() -> None:
    args = parse_args()
    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=1, pp=1, tp=1, cp=runtime.world_size, ep=1)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        if args.seq_len % len(groups.cp_ranks) != 0:
            raise ValueError("seq_len must be divisible by cp size for this ring test")

        torch.manual_seed(args.seed)
        shape = (
            args.batch_size,
            args.n_heads,
            args.seq_len,
            args.head_dim,
        )
        q_full = torch.randn(shape, device=runtime.device)
        k_full = torch.randn(shape, device=runtime.device)
        v_full = torch.randn(shape, device=runtime.device)

        q_local = shard_sequence(q_full, groups, seq_dim=2).contiguous()
        k_local = shard_sequence(k_full, groups, seq_dim=2).contiguous()
        v_local = shard_sequence(v_full, groups, seq_dim=2).contiguous()

        gathered_q = gather_sequence(q_local, groups, seq_dim=2)
        gather_diff = (gathered_q - q_full).abs().max()

        scale = 1.0 / math.sqrt(args.head_dim)
        expected_full = full_causal_attention(
            q_full,
            k_full,
            v_full,
            scale=scale,
        )
        expected_local = shard_sequence(expected_full, groups, seq_dim=2)

        actual_local = ring_causal_attention(
            q_local,
            k_local,
            v_local,
            groups,
            sequence_length=args.seq_len,
            scale=scale,
        )

        attention_diff = (actual_local - expected_local).abs().max()
        global_gather_diff = gather_diff.clone()
        global_attention_diff = attention_diff.clone()

        if runtime.is_distributed:
            dist.all_reduce(
                global_gather_diff,
                op=dist.ReduceOp.MAX,
                group=groups.cp_group,
            )
            dist.all_reduce(
                global_attention_diff,
                op=dist.ReduceOp.MAX,
                group=groups.cp_group,
            )

        global_max_diff = torch.maximum(global_gather_diff, global_attention_diff)
        passed = bool(global_max_diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)

        if runtime.is_distributed:
            dist.all_reduce(
                passed_tensor,
                op=dist.ReduceOp.MIN,
                group=groups.cp_group,
            )
        global_passed = bool(passed_tensor.item())

        shard = context_parallel_range(args.seq_len, groups)
        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  cp_ranks={groups.cp_ranks}")
                print(f"  cp_rank={shard.cp_rank}")
                print(f"  shard=[{shard.start}, {shard.end})")
                print(f"  q_local_shape={tuple(q_local.shape)}")
                print(f"  local_gather_diff={float(gather_diff.item())}")
                print(f"  local_attention_diff={float(attention_diff.item())}")
                print(f"  global_gather_diff={float(global_gather_diff.item())}")
                print(f"  global_attention_diff={float(global_attention_diff.item())}")
                print(f"  global_max_diff={float(global_max_diff.item())}")
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
