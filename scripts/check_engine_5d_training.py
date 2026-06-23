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

from meshtrain.engine.config import (  # noqa: E402
    CheckpointConfig,
    DataConfig,
    EngineConfig,
    MoEConfig,
    OptimizerConfig,
    ParallelismConfig,
    PrecisionConfig,
    TrainingConfig,
)
from meshtrain.engine.trainer import EngineTrainer  # noqa: E402
from meshtrain.model.standard_transformer import TransformerConfig  # noqa: E402
from meshtrain.parallelism.data_parallel.data_parallelism import (  # noqa: E402
    check_replicas_match,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"),
    )

    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)

    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)

    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        parallelism=ParallelismConfig(
            dp=args.dp,
            pp=args.pp,
            tp=args.tp,
            cp=args.cp,
            ep=args.ep,
            pp_schedule="gpipe" if args.pp > 1 else "none",
        ),
        model=TransformerConfig(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            mlp_hidden_dim=args.mlp_hidden_dim,
            tie_embeddings=False,
        ),
        training=TrainingConfig(
            max_steps=args.steps,
            global_batch_size=args.global_batch_size,
            microbatch_size=args.microbatch_size,
            seed=args.seed,
            log_every=1,
            checkpoint_every=max(args.steps + 1, 1),
        ),
        optimizer=OptimizerConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        data=DataConfig(
            train_path=str(args.train_path),
            seq_len=args.seq_len,
        ),
        precision=PrecisionConfig(
            dtype="fp32",
            autocast=False,
            grad_scaler=False,
        ),
        checkpoint=CheckpointConfig(
            output_dir="tmp/engine_5d_training_checkpoints",
            save_optimizer=True,
        ),
        moe=MoEConfig(
            enabled=True,
            num_experts=args.num_experts,
        ),
    )


def module_for_replica_check(trainer: EngineTrainer) -> torch.nn.Module:
    if trainer.uses_pipeline_parallel:
        assert trainer.pipeline_stage is not None
        return trainer.pipeline_stage

    assert trainer.model is not None
    return trainer.model


def progress(trainer: EngineTrainer, message: str, enabled: bool) -> None:
    if not enabled:
        return
    if trainer.context is None:
        print(f"rank=? {message}", flush=True)
        return
    print(f"rank={trainer.context.runtime.rank} {message}", flush=True)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    trainer = EngineTrainer(config)
    losses: list[float] = []

    try:
        progress(trainer, "setup:start", args.progress)
        trainer.setup()
        assert trainer.context is not None
        progress(trainer, "setup:done", args.progress)

        runtime = trainer.context.runtime
        if runtime.world_size != config.parallelism.world_size:
            raise ValueError(
                f"parallelism requires world_size={config.parallelism.world_size}, "
                f"got runtime world_size={runtime.world_size}"
            )

        for _ in range(args.steps):
            progress(trainer, f"step:{trainer.state.step}:start", args.progress)
            metrics = trainer.train_step()
            progress(
                trainer,
                f"step:{metrics.step}:done loss={metrics.loss:.6f}",
                args.progress,
            )
            if not math.isfinite(metrics.loss):
                raise RuntimeError(f"non-finite loss at step {metrics.step}: {metrics.loss}")
            losses.append(metrics.loss)

        replica_check = check_replicas_match(
            module_for_replica_check(trainer),
            trainer.context.groups,
            atol=args.atol,
        )
        passed = replica_check.passed and all(math.isfinite(loss) for loss in losses)
        passed_tensor = torch.tensor(int(passed), device=runtime.device)
        if runtime.is_distributed:
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        global_passed = bool(passed_tensor.item())

        for rank in range(runtime.world_size):
            if runtime.rank == rank:
                coord = trainer.context.placement.coord
                groups = trainer.context.groups
                print(f"rank={runtime.rank}")
                print(
                    "  coord="
                    f"dp{coord.dp}/pp{coord.pp}/tp{coord.tp}/cp{coord.cp}/ep{coord.ep}"
                )
                print(f"  dp_group={groups.dp_ranks}")
                print(f"  pp_group={groups.pp_ranks}")
                print(f"  tp_group={groups.tp_ranks}")
                print(f"  cp_group={groups.cp_ranks}")
                print(f"  ep_group={groups.ep_ranks}")
                if trainer.pipeline_stage is not None:
                    print(
                        "  layers="
                        f"[{trainer.pipeline_stage.layer_start}, {trainer.pipeline_stage.layer_end})"
                    )
                print(f"  losses={[round(loss, 6) for loss in losses]}")
                print(f"  replica_max_abs_diff={replica_check.max_abs_diff}")
            if runtime.is_distributed:
                dist.barrier()

        if runtime.rank == 0:
            print(f"passed: {global_passed}")
        if not global_passed:
            raise SystemExit(1)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

