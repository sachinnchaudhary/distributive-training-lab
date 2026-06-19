from __future__ import annotations

import argparse
import copy
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
from meshtrain.data.dataloader import DataLoader
from meshtrain.data.dataset import TokenShardDataset
from meshtrain.data.packing import CausalLMPacker
from meshtrain.data.sampler import DPSampler
from meshtrain.model.standard_transformer import TransformerConfig, TransformerLM
from meshtrain.parallelism.data_parallel import (
    ZeroAdamW,
    ZeroConfig,
    ZeroStage,
    broadcast_parameters,
    check_replicas_match,
    sync_loss_stats,
)
from meshtrain.training.loss import next_token_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data_downloads/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"),
    )
    parser.add_argument("--zero-stage", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def build_loader(
    *,
    train_path: Path,
    seq_len: int,
    global_batch_size: int,
    dp_rank: int,
    dp_size: int,
    device: torch.device,
) -> DataLoader:
    if not train_path.exists():
        raise FileNotFoundError(
            f"train shard not found: {train_path}. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first, "
            "or pass --train-path."
        )

    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=seq_len)
    sampler = DPSampler(
        num_examples=len(packer),
        global_batch_size=global_batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    return DataLoader(packer=packer, sampler=sampler, device=device)


def build_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    return TransformerLM(config).to(device)


def build_zero_config(args: argparse.Namespace) -> ZeroConfig:
    return ZeroConfig(
        stage=ZeroStage(args.zero_stage),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def build_reference_optimizer(
    model: TransformerLM,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def reference_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batch,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(batch.input_ids)
    loss_out = next_token_loss(logits, batch.target_ids)
    loss_out.loss.backward()
    optimizer.step()
    return loss_out.loss.detach()


def zero_step(
    model: TransformerLM,
    optimizer: ZeroAdamW,
    batch,
    groups,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer.gather_parameters()

    logits = model(batch.input_ids)
    loss_out = next_token_loss(logits, batch.target_ids)
    global_stats = sync_loss_stats(loss_out.loss_sum, loss_out.num_tokens, groups)

    backward_loss = loss_out.loss_sum / global_stats.num_tokens.clamp_min(1)
    backward_loss.backward()

    optimizer.sync_gradients()
    optimizer.step()
    return global_stats.loss.detach()


def max_diff_from_reference(
    model: TransformerLM,
    reference_model: TransformerLM | None,
    *,
    src_rank: int,
    group: dist.ProcessGroup | None,
) -> float:
    device = next(model.parameters()).device
    max_diff = torch.zeros((), device=device)
    reference_params = dict(reference_model.named_parameters()) if reference_model else {}
    distributed = dist.is_available() and dist.is_initialized()

    for name, parameter in model.named_parameters():
        reference = torch.empty_like(parameter.detach())
        if reference_model is not None:
            reference.copy_(reference_params[name].detach())

        if distributed:
            dist.broadcast(reference, src=src_rank, group=group)
        elif reference_model is None:
            raise RuntimeError("reference_model is required when distributed is not initialized")

        diff = (parameter.detach() - reference).abs().max()
        max_diff = torch.maximum(max_diff, diff)

    if distributed:
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX, group=group)
    return float(max_diff.item())


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
        model = build_model(args, runtime.device)
        broadcast_parameters(model, groups)

        initial_state = copy.deepcopy(model.state_dict()) if runtime.rank == 0 else None
        zero_optimizer = ZeroAdamW(model, groups, build_zero_config(args))

        loader = build_loader(
            train_path=args.train_path,
            seq_len=args.seq_len,
            global_batch_size=args.global_batch_size,
            dp_rank=mesh.coord_for_rank(runtime.rank).dp,
            dp_size=runtime.world_size,
            device=runtime.device,
        )
        local_batch = loader.get_batch(0)

        reference_model = None
        reference_loss = None
        if runtime.rank == 0:
            reference_model = build_model(args, runtime.device)
            reference_model.load_state_dict(initial_state)
            reference_optimizer = build_reference_optimizer(reference_model, args)
            reference_loader = build_loader(
                train_path=args.train_path,
                seq_len=args.seq_len,
                global_batch_size=args.global_batch_size,
                dp_rank=0,
                dp_size=1,
                device=runtime.device,
            )
            reference_batch = reference_loader.get_batch(0)
            reference_loss = reference_step(
                reference_model,
                reference_optimizer,
                reference_batch,
            )

        zero_loss = zero_step(model, zero_optimizer, local_batch, groups)
        replica_check = check_replicas_match(model, groups, atol=args.atol)
        reference_diff = max_diff_from_reference(
            model,
            reference_model,
            src_rank=groups.dp_ranks[0],
            group=groups.dp_group,
        )

        passed = replica_check.passed and reference_diff <= args.atol
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                print(f"rank={runtime.rank}")
                print(f"  zero_stage={args.zero_stage}")
                print(f"  sample_ids={local_batch.sample_ids.tolist()}")
                print(f"  local_batch_range={local_batch.local_batch_range}")
                print(f"  owned_parameters={zero_optimizer.local_owned_parameter_names()}")
                print(f"  local_zero_state_numel={zero_optimizer.local_state_numel()}")
                print(f"  local_grad_shard_numel={zero_optimizer.local_grad_shard_numel()}")
                print(f"  local_param_shard_numel={zero_optimizer.local_param_shard_numel()}")
                print(f"  zero_global_loss={float(zero_loss)}")
                print(f"  replica_max_abs_diff={replica_check.max_abs_diff}")
                print(f"  reference_max_abs_diff={reference_diff}")
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
