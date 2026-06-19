from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--reshard-after-forward", action="store_true", default=True)
    return parser.parse_args()


def broadcast_module_parameters(module: nn.Module, groups) -> None:
    if groups.dp_group is None or len(groups.dp_ranks) <= 1:
        return

    src_rank = groups.dp_ranks[0]
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src_rank, group=groups.dp_group)


def build_reference_optimizer(
    module: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        module.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def mse_loss_sum(output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_sum = F.mse_loss(output, target, reduction="sum")
    numel = torch.tensor(target.numel(), device=target.device)
    return loss_sum, numel


def reference_step(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    module.train()
    optimizer.zero_grad(set_to_none=True)

    output = module(x)
    loss_sum, numel = mse_loss_sum(output, target)
    loss = loss_sum / numel.clamp_min(1)
    loss.backward()
    optimizer.step()
    return loss.detach()


def fsdp_step(
    module: FullyShardedModule,
    x: torch.Tensor,
    target: torch.Tensor,
    groups,
) -> torch.Tensor:
    module.train()
    module.zero_grad(set_to_none=True)

    output = module(x)
    loss_sum, numel = mse_loss_sum(output, target)

    global_loss_sum = loss_sum.detach().clone()
    global_numel = numel.detach().clone()
    if groups.dp_group is not None and len(groups.dp_ranks) > 1:
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=groups.dp_group)
        dist.all_reduce(global_numel, op=dist.ReduceOp.SUM, group=groups.dp_group)

    backward_loss = loss_sum / global_numel.clamp_min(1)
    backward_loss.backward()

    module.sync_gradients()
    module.step()

    return (global_loss_sum / global_numel.clamp_min(1)).detach()


def max_param_diff(module_a: nn.Module, module_b: nn.Module) -> float:
    max_diff = 0.0
    for param_a, param_b in zip(module_a.parameters(), module_b.parameters()):
        diff = (param_a.detach() - param_b.detach()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


def main() -> None:
    args = parse_args()
    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=runtime.world_size, pp=1, tp=1, cp=1, ep=1)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        if args.global_batch_size % runtime.world_size != 0:
            raise ValueError(
                f"global_batch_size {args.global_batch_size} must divide evenly "
                f"by dp_size {runtime.world_size}"
            )

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
            FSDPConfig(
                reshard_after_forward=args.reshard_after_forward,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            ),
        ).to(runtime.device)

        torch.manual_seed(args.seed + 1)
        x = torch.randn(
            args.global_batch_size,
            args.in_features,
            device=runtime.device,
        )
        target = torch.randn(
            args.global_batch_size,
            args.out_features,
            device=runtime.device,
        )

        local_batch_size = args.global_batch_size // runtime.world_size
        local_start = runtime.rank * local_batch_size
        local_end = local_start + local_batch_size
        local_x = x[local_start:local_end]
        local_target = target[local_start:local_end]

        reference_loss = None
        if runtime.rank == 0:
            reference_optimizer = build_reference_optimizer(reference, args)
            reference_loss = reference_step(reference, reference_optimizer, x, target)

        fsdp_loss = fsdp_step(fsdp_module, local_x, local_target, groups)

        reference_diff = torch.zeros((), device=runtime.device)
        if runtime.rank == 0:
            reference_diff.fill_(max_param_diff(fsdp_module.module, reference))
        if runtime.is_distributed:
            dist.broadcast(reference_diff, src=groups.dp_ranks[0], group=groups.dp_group)

        passed = bool(reference_diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  local_batch_range=({local_start}, {local_end})")
                print(f"  local_input_shape={tuple(local_x.shape)}")
                print(f"  local_param_shard_numel={fsdp_module.local_param_shard.numel()}")
                print(f"  local_grad_shard_numel={fsdp_module.local_grad_shard_numel()}")
                print(f"  local_state_numel={fsdp_module.local_state_numel()}")
                print(f"  total_module_numel={fsdp_module.total_numel}")
                print(f"  fsdp_global_loss={float(fsdp_loss)}")
                print(f"  reference_max_abs_diff={float(reference_diff.item())}")
                if reference_loss is not None:
                    print(f"  reference_loss={float(reference_loss)}")
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
