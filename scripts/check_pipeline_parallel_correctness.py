from __future__ import annotations

import argparse
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
from meshtrain.parallelism.pipeline_parallel.schedules import (
    gpipe_forward_backward,
    interleaved_one_forward_one_backward,
    one_forward_one_backward,
    pipeline_forward,
    pipeline_forward_backward,
)
from meshtrain.parallelism.pipeline_parallel.stage import (
    build_interleaved_pipeline_stages_from_layers,
    build_pipeline_stage_from_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-linear-layers", type=int, default=4)
    parser.add_argument("--num-microbatches", type=int, default=2)
    parser.add_argument("--virtual-stages-per-rank", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def build_reference_model(hidden_dim: int, num_linear_layers: int) -> nn.Sequential:
    if num_linear_layers < 1:
        raise ValueError("num_linear_layers must be >= 1")

    layers: list[nn.Module] = []
    for layer_index in range(num_linear_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        if layer_index != num_linear_layers - 1:
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


def broadcast_module_parameters(module: nn.Module, group, src_rank: int) -> None:
    if group is None:
        return

    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src_rank, group=group)


def max_local_grad_diff(stage, reference: nn.Sequential) -> torch.Tensor:
    max_diff = torch.zeros((), device=next(reference.parameters()).device)

    reference_layers = list(reference.children())
    for local_layer_index, stage_layer in enumerate(stage.layers):
        reference_layer_index = stage.layer_start + local_layer_index
        reference_layer = reference_layers[reference_layer_index]

        for stage_parameter, reference_parameter in zip(
            stage_layer.parameters(),
            reference_layer.parameters(),
        ):
            if stage_parameter.grad is None and reference_parameter.grad is None:
                continue
            if stage_parameter.grad is None or reference_parameter.grad is None:
                return torch.tensor(float("inf"), device=max_diff.device)

            diff = (stage_parameter.grad - reference_parameter.grad).abs().max()
            max_diff = torch.maximum(max_diff, diff)

    return max_diff


def max_local_grad_diff_for_stages(
    stages,
    reference: nn.Sequential,
) -> torch.Tensor:
    max_diff = torch.zeros((), device=next(reference.parameters()).device)

    for stage in stages:
        max_diff = torch.maximum(max_diff, max_local_grad_diff(stage, reference))

    return max_diff


def main() -> None:
    args = parse_args()
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError("batch_size must be divisible by num_microbatches")

    runtime = init_runtime()

    try:
        dims = ParallelDims(dp=1, pp=runtime.world_size, tp=1, cp=1, ep=1)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)

        torch.manual_seed(args.seed)
        reference = build_reference_model(
            hidden_dim=args.hidden_dim,
            num_linear_layers=args.num_linear_layers,
        ).to(runtime.device)
        pipeline_model = build_reference_model(
            hidden_dim=args.hidden_dim,
            num_linear_layers=args.num_linear_layers,
        ).to(runtime.device)
        pipeline_model.load_state_dict(reference.state_dict())
        interleaved_model = build_reference_model(
            hidden_dim=args.hidden_dim,
            num_linear_layers=args.num_linear_layers,
        ).to(runtime.device)
        interleaved_model.load_state_dict(reference.state_dict())

        if runtime.is_distributed:
            broadcast_module_parameters(reference, groups.pp_group, groups.pp_ranks[0])
            broadcast_module_parameters(pipeline_model, groups.pp_group, groups.pp_ranks[0])
            broadcast_module_parameters(interleaved_model, groups.pp_group, groups.pp_ranks[0])

        stage = build_pipeline_stage_from_layers(
            list(pipeline_model.children()),
            groups,
        ).to(runtime.device)
        interleaved_stages = build_interleaved_pipeline_stages_from_layers(
            list(interleaved_model.children()),
            groups,
            virtual_stages_per_rank=args.virtual_stages_per_rank,
        )
        for interleaved_stage in interleaved_stages:
            interleaved_stage.to(runtime.device)

        torch.manual_seed(args.seed + 1)
        x = torch.randn(
            args.batch_size,
            args.seq_len,
            args.hidden_dim,
            device=runtime.device,
        )

        target = torch.randn_like(x)

        expected = reference(x)
        actual = pipeline_forward(
            stage,
            x if stage.is_first else None,
            activation_shape=tuple(x.shape),
            dtype=x.dtype,
            device=runtime.device,
        )

        if stage.is_last:
            local_diff = (actual - expected).abs().max()
        else:
            local_diff = torch.zeros((), device=runtime.device)

        global_diff = local_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(global_diff, op=dist.ReduceOp.MAX, group=groups.pp_group)

        reference.zero_grad(set_to_none=True)
        pipeline_model.zero_grad(set_to_none=True)

        reference_output = reference(x)
        reference_loss = F.mse_loss(reference_output, target)
        reference_loss.backward()

        pipeline_loss = pipeline_forward_backward(
            stage,
            x if stage.is_first else None,
            activation_shape=tuple(x.shape),
            dtype=x.dtype,
            device=runtime.device,
            loss_fn=(lambda output: F.mse_loss(output, target)) if stage.is_last else None,
        )

        if stage.is_last:
            loss_diff = (pipeline_loss - reference_loss.detach()).abs()
        else:
            loss_diff = torch.zeros((), device=runtime.device)
        grad_diff = max_local_grad_diff(stage, reference)

        global_loss_diff = loss_diff.clone()
        global_grad_diff = grad_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(global_loss_diff, op=dist.ReduceOp.MAX, group=groups.pp_group)
            dist.all_reduce(global_grad_diff, op=dist.ReduceOp.MAX, group=groups.pp_group)

        reference.zero_grad(set_to_none=True)
        pipeline_model.zero_grad(set_to_none=True)

        reference_output = reference(x)
        reference_loss = F.mse_loss(reference_output, target)
        reference_loss.backward()

        input_microbatches = list(x.chunk(args.num_microbatches, dim=0))
        target_microbatches = list(target.chunk(args.num_microbatches, dim=0))
        microbatch_shape = tuple(input_microbatches[0].shape)

        gpipe_losses = gpipe_forward_backward(
            stage,
            input_microbatches if stage.is_first else None,
            num_microbatches=args.num_microbatches,
            activation_shape=microbatch_shape,
            dtype=x.dtype,
            device=runtime.device,
            loss_fn=(
                lambda output, microbatch_id: F.mse_loss(
                    output,
                    target_microbatches[microbatch_id],
                )
                / args.num_microbatches
            )
            if stage.is_last
            else None,
        )

        if stage.is_last:
            gpipe_loss = torch.stack(gpipe_losses).sum()
            gpipe_loss_diff = (gpipe_loss - reference_loss.detach()).abs()
        else:
            gpipe_loss_diff = torch.zeros((), device=runtime.device)
        gpipe_grad_diff = max_local_grad_diff(stage, reference)

        global_gpipe_loss_diff = gpipe_loss_diff.clone()
        global_gpipe_grad_diff = gpipe_grad_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(
                global_gpipe_loss_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )
            dist.all_reduce(
                global_gpipe_grad_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )

        reference.zero_grad(set_to_none=True)
        pipeline_model.zero_grad(set_to_none=True)

        reference_output = reference(x)
        reference_loss = F.mse_loss(reference_output, target)
        reference_loss.backward()

        one_f_one_b_losses = one_forward_one_backward(
            stage,
            input_microbatches if stage.is_first else None,
            num_microbatches=args.num_microbatches,
            activation_shape=microbatch_shape,
            dtype=x.dtype,
            device=runtime.device,
            loss_fn=(
                lambda output, microbatch_id: F.mse_loss(
                    output,
                    target_microbatches[microbatch_id],
                )
                / args.num_microbatches
            )
            if stage.is_last
            else None,
        )

        if stage.is_last:
            one_f_one_b_loss = torch.stack(one_f_one_b_losses).sum()
            one_f_one_b_loss_diff = (one_f_one_b_loss - reference_loss.detach()).abs()
        else:
            one_f_one_b_loss_diff = torch.zeros((), device=runtime.device)
        one_f_one_b_grad_diff = max_local_grad_diff(stage, reference)

        global_one_f_one_b_loss_diff = one_f_one_b_loss_diff.clone()
        global_one_f_one_b_grad_diff = one_f_one_b_grad_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(
                global_one_f_one_b_loss_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )
            dist.all_reduce(
                global_one_f_one_b_grad_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )

        reference.zero_grad(set_to_none=True)
        interleaved_model.zero_grad(set_to_none=True)

        reference_output = reference(x)
        reference_loss = F.mse_loss(reference_output, target)
        reference_loss.backward()

        interleaved_losses = interleaved_one_forward_one_backward(
            interleaved_stages,
            input_microbatches if groups.rank == groups.pp_ranks[0] else None,
            num_microbatches=args.num_microbatches,
            activation_shape=microbatch_shape,
            dtype=x.dtype,
            device=runtime.device,
            loss_fn=(
                lambda output, microbatch_id: F.mse_loss(
                    output,
                    target_microbatches[microbatch_id],
                )
                / args.num_microbatches
            )
            if any(
                stage.global_virtual_stage_index == stage.num_virtual_stages - 1
                for stage in interleaved_stages
            )
            else None,
        )

        owns_last_virtual_stage = any(
            stage.global_virtual_stage_index == stage.num_virtual_stages - 1
            for stage in interleaved_stages
        )
        if owns_last_virtual_stage:
            interleaved_loss = torch.stack(interleaved_losses).sum()
            interleaved_loss_diff = (interleaved_loss - reference_loss.detach()).abs()
        else:
            interleaved_loss_diff = torch.zeros((), device=runtime.device)
        interleaved_grad_diff = max_local_grad_diff_for_stages(
            interleaved_stages,
            reference,
        )

        global_interleaved_loss_diff = interleaved_loss_diff.clone()
        global_interleaved_grad_diff = interleaved_grad_diff.clone()
        if runtime.is_distributed:
            dist.all_reduce(
                global_interleaved_loss_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )
            dist.all_reduce(
                global_interleaved_grad_diff,
                op=dist.ReduceOp.MAX,
                group=groups.pp_group,
            )

        global_max_diff = torch.maximum(
            global_diff,
            torch.maximum(
                torch.maximum(global_loss_diff, global_grad_diff),
                torch.maximum(
                    torch.maximum(global_gpipe_loss_diff, global_gpipe_grad_diff),
                    torch.maximum(
                        torch.maximum(
                            global_one_f_one_b_loss_diff,
                            global_one_f_one_b_grad_diff,
                        ),
                        torch.maximum(
                            global_interleaved_loss_diff,
                            global_interleaved_grad_diff,
                        ),
                    ),
                ),
            ),
        )

        passed = bool(global_max_diff.item() <= args.atol)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN, group=groups.pp_group)
        global_passed = bool(passed_tensor.item())

        info = stage.info()
        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  pp_ranks={groups.pp_ranks}")
                print(f"  stage_index={info.stage_index}")
                print(f"  layers=[{info.layer_start}, {info.layer_end})")
                print(f"  is_first={stage.is_first}")
                print(f"  is_last={stage.is_last}")
                print(f"  num_microbatches={args.num_microbatches}")
                print(f"  virtual_stages_per_rank={args.virtual_stages_per_rank}")
                print("  virtual_stages:")
                for virtual_stage in interleaved_stages:
                    virtual_info = virtual_stage.virtual_info()
                    print(
                        "    "
                        f"local={virtual_info.virtual_stage_index} "
                        f"global={virtual_info.global_virtual_stage_index} "
                        f"layers=[{virtual_info.layer_start}, {virtual_info.layer_end})"
                    )
                if stage.is_last:
                    print(f"  pipeline_output_shape={tuple(actual.shape)}")
                    print(f"  local_forward_max_diff={float(local_diff.item())}")
                    print(f"  local_loss_diff={float(loss_diff.item())}")
                    print(f"  local_gpipe_loss_diff={float(gpipe_loss_diff.item())}")
                    print(f"  local_1f1b_loss_diff={float(one_f_one_b_loss_diff.item())}")
                if owns_last_virtual_stage:
                    print(f"  local_interleaved_loss_diff={float(interleaved_loss_diff.item())}")
                print(f"  local_grad_max_diff={float(grad_diff.item())}")
                print(f"  local_gpipe_grad_max_diff={float(gpipe_grad_diff.item())}")
                print(f"  local_1f1b_grad_max_diff={float(one_f_one_b_grad_diff.item())}")
                print(f"  local_interleaved_grad_max_diff={float(interleaved_grad_diff.item())}")
                print(f"  global_forward_max_diff={float(global_diff.item())}")
                print(f"  global_loss_diff={float(global_loss_diff.item())}")
                print(f"  global_grad_max_diff={float(global_grad_diff.item())}")
                print(f"  global_gpipe_loss_diff={float(global_gpipe_loss_diff.item())}")
                print(f"  global_gpipe_grad_max_diff={float(global_gpipe_grad_diff.item())}")
                print(f"  global_1f1b_loss_diff={float(global_one_f_one_b_loss_diff.item())}")
                print(f"  global_1f1b_grad_max_diff={float(global_one_f_one_b_grad_diff.item())}")
                print(f"  global_interleaved_loss_diff={float(global_interleaved_loss_diff.item())}")
                print(f"  global_interleaved_grad_max_diff={float(global_interleaved_grad_diff.item())}")
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
