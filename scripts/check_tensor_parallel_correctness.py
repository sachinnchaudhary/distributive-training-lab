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
from meshtrain.model.standard_transformer import CausalSelfAttention, MLP, TransformerConfig
from meshtrain.parallelism.tensor_parallel.attention import TensorParallelSelfAttention
from meshtrain.parallelism.tensor_parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from meshtrain.parallelism.tensor_parallel.mlp import TensorParallelMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["linear", "mlp", "attention", "all"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def broadcast_linear(linear: nn.Linear, groups) -> None:
    if groups.tp_group is None or len(groups.tp_ranks) <= 1:
        return

    src_rank = groups.tp_ranks[0]
    dist.broadcast(linear.weight.data, src=src_rank, group=groups.tp_group)
    if linear.bias is not None:
        dist.broadcast(linear.bias.data, src=src_rank, group=groups.tp_group)


def broadcast_module_parameters(module: nn.Module, groups) -> None:
    if groups.tp_group is None or len(groups.tp_ranks) <= 1:
        return

    src_rank = groups.tp_ranks[0]
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src_rank, group=groups.tp_group)


def run_linear_case(args: argparse.Namespace, groups, device: torch.device) -> dict[str, float]:
    torch.manual_seed(args.seed)
    reference = nn.Linear(
        args.in_features,
        args.out_features,
        bias=True,
    ).to(device)
    broadcast_linear(reference, groups)

    column = ColumnParallelLinear(
        args.in_features,
        args.out_features,
        groups,
        bias=True,
        gather_output=True,
    ).to(device)
    row = RowParallelLinear(
        args.in_features,
        args.out_features,
        groups,
        bias=True,
        input_is_parallel=False,
    ).to(device)

    column.load_from_linear(reference)
    row.load_from_linear(reference)

    torch.manual_seed(args.seed + 1)
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.in_features,
        device=device,
    )

    expected = reference(x)
    column_output = column(x)
    row_output = row(x)

    return {
        "column_max_diff": float((column_output - expected).abs().max().item()),
        "row_max_diff": float((row_output - expected).abs().max().item()),
    }


def run_mlp_case(args: argparse.Namespace, groups, device: torch.device) -> dict[str, float]:
    torch.manual_seed(args.seed)
    config = TransformerConfig(
        vocab_size=1024,
        seq_len=args.seq_len,
        dim=args.in_features,
        n_layers=1,
        n_heads=1,
        mlp_hidden_dim=args.hidden_dim,
    )
    reference = MLP(config).to(device)
    broadcast_module_parameters(reference, groups)

    tp_mlp = TensorParallelMLP(
        dim=args.in_features,
        hidden_dim=args.hidden_dim,
        groups=groups,
        bias=False,
    ).to(device)
    tp_mlp.load_from_mlp(reference)

    torch.manual_seed(args.seed + 1)
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.in_features,
        device=device,
    )

    expected = reference(x)
    actual = tp_mlp(x)

    return {
        "mlp_max_diff": float((actual - expected).abs().max().item()),
    }


def run_attention_case(
    args: argparse.Namespace,
    groups,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(args.seed)
    config = TransformerConfig(
        vocab_size=1024,
        seq_len=args.seq_len,
        dim=args.in_features,
        n_layers=1,
        n_heads=args.n_heads,
        mlp_hidden_dim=args.hidden_dim,
        dropout=0.0,
    )
    reference = CausalSelfAttention(config).to(device)
    broadcast_module_parameters(reference, groups)

    tp_attention = TensorParallelSelfAttention(
        dim=args.in_features,
        n_heads=args.n_heads,
        groups=groups,
        dropout=0.0,
        bias=False,
    ).to(device)
    tp_attention.load_from_attention(reference)

    torch.manual_seed(args.seed + 1)
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.in_features,
        device=device,
    )

    expected = reference(x)
    actual = tp_attention(x)

    return {
        "attention_max_diff": float((actual - expected).abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=1, pp=1, tp=runtime.world_size, cp=1, ep=1)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        if args.case not in {"linear", "mlp", "attention", "all"}:
            raise NotImplementedError(f"case {args.case!r} is not implemented yet")

        results = {}
        if args.case in {"linear", "all"}:
            results.update(run_linear_case(args, groups, runtime.device))
        if args.case in {"mlp", "all"}:
            results.update(run_mlp_case(args, groups, runtime.device))
        if args.case in {"attention", "all"}:
            results.update(run_attention_case(args, groups, runtime.device))

        max_diff = torch.tensor(max(results.values()), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(max_diff, op=dist.ReduceOp.MAX, group=groups.tp_group)

        passed = bool(max_diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN, group=groups.tp_group)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  case={args.case}")
                print(f"  tp_ranks={groups.tp_ranks}")
                for name, value in results.items():
                    print(f"  {name}={value}")
                print(f"  global_max_diff={float(max_diff.item())}")
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
