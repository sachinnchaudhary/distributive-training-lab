from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from meshtrain.data.dataloader import DataLoader
from meshtrain.data.dataset import TokenShardDataset
from meshtrain.data.packing import CausalLMPacker
from meshtrain.data.sampler import DPSampler
from meshtrain.engine.activation_checkpoint import apply_activation_checkpointing
from meshtrain.engine.builder import (
    EngineContext,
    build_engine_context,
    shutdown_engine_context,
)
from meshtrain.engine.checkpoint import load_checkpoint, save_checkpoint
from meshtrain.engine.config import EngineConfig
from meshtrain.engine.precision import (
    autocast_context,
    create_grad_scaler,
    maybe_cast_model,
)
from meshtrain.model.moe_transformer import (
    MoETransformerLM,
    MoETransformerPipelineStage,
)
from meshtrain.model.standard_transformer import TransformerLM
from meshtrain.parallelism.context_parallelism import shard_sequence
from meshtrain.parallelism.data_parallel.data_parallelism import (
    broadcast_parameters,
    sync_gradients,
    sync_loss_stats,
)
from meshtrain.parallelism.pipeline_parallel.schedules import gpipe_forward_backward
from meshtrain.parallelism.pipeline_parallel.stage import (
    PipelineStage,
    build_transformer_pipeline_stage,
)
from meshtrain.parallelism.tensor_parallel.transformer import TensorParallelTransformerLM
from meshtrain.training.loss import next_token_loss


@dataclass(frozen=True)
class TrainMetrics:
    step: int
    global_batch_id: int
    loss: float
    loss_sum: float
    num_tokens: int


@dataclass(frozen=True)
class TrainerState:
    step: int
    global_batch_id: int


class EngineTrainer:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.context: EngineContext | None = None
        self.model: nn.Module | None = None
        self.pipeline_stage: PipelineStage | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.dataloader: DataLoader | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.state = TrainerState(
            step=0,
            global_batch_id=config.data.global_batch_id_start,
        )

    @property
    def runtime_rank(self) -> int:
        return 0 if self.context is None else self.context.runtime.rank

    @property
    def is_rank_zero(self) -> bool:
        return self.runtime_rank == 0

    @property
    def uses_pipeline_parallel(self) -> bool:
        return self.config.parallelism.pp > 1

    def _train_module(self) -> nn.Module:
        if self.uses_pipeline_parallel:
            if self.pipeline_stage is None:
                raise RuntimeError("pipeline stage has not been built")
            return self.pipeline_stage

        if self.model is None:
            raise RuntimeError("model has not been built")
        return self.model

    def _debug_step(self, message: str) -> None:
        if os.environ.get("MESHTRAIN_ENGINE_DEBUG", "0") != "1":
            return
        rank = "?" if self.context is None else self.context.runtime.rank
        print(f"rank={rank} engine:{message}", flush=True)

    def _validate_supported_parallelism(self) -> None:
        parallelism = self.config.parallelism

        if parallelism.zero_stage != 0:
            raise NotImplementedError("ZeRO is not wired into EngineTrainer yet")
        if parallelism.fsdp:
            raise NotImplementedError("FSDP is not wired into EngineTrainer yet")

        

    def setup(self) -> None:
        if self.context is not None:
            return

        self._validate_supported_parallelism()
        self.context = build_engine_context(self.config)

        try:
            self._seed()

            if self.uses_pipeline_parallel:
                self._setup_pipeline()
            else:
                self._setup_single_stage()

            self.scaler = create_grad_scaler(
                self.config.precision,
                device=self.context.runtime.device,
            )
            if self.scaler is not None and self.uses_pipeline_parallel:
                raise NotImplementedError("fp16 GradScaler is not wired for GPipe yet")

            self.dataloader = self._build_dataloader()

            if self.config.checkpoint.resume_from is not None:
                if self.uses_pipeline_parallel:
                    raise NotImplementedError("PP checkpoint resume is not wired yet")

                loaded_step = load_checkpoint(
                    model=self._train_module(),
                    optimizer=self.optimizer,
                    path=self.config.checkpoint.resume_from,
                    map_location=self.context.runtime.device,
                    load_optimizer=self.config.checkpoint.save_optimizer,
                )
                self.state = TrainerState(
                    step=loaded_step,
                    global_batch_id=self.config.data.global_batch_id_start + loaded_step,
                )
            else:
                broadcast_parameters(self._train_module(), self.context.groups)
        except Exception:
            self.close()
            raise

    def _seed(self) -> None:
        torch.manual_seed(self.config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.training.seed)

    def _setup_single_stage(self) -> None:
        self.model = self._build_model()
        self.optimizer = self._build_optimizer(self.model)

    def _setup_pipeline(self) -> None:
        assert self.context is not None

        if self._uses_moe_transformer:
            stage = self._build_moe_pipeline_stage().to(self.context.runtime.device)
        else:
            stage = build_transformer_pipeline_stage(
                self.config.model,
                self.context.groups,
            ).to(self.context.runtime.device)
        stage = maybe_cast_model(stage, self.config.precision)

        self.pipeline_stage = stage
        self.optimizer = self._build_optimizer(stage)

    @property
    def _uses_moe_transformer(self) -> bool:
        parallelism = self.config.parallelism
        return self.config.moe.enabled or parallelism.cp > 1 or parallelism.ep > 1

    def _build_model(self) -> nn.Module:
        assert self.context is not None

        if self._uses_moe_transformer:
            model = MoETransformerLM(
                self.config.model,
                self.context.groups,
                num_experts=self.config.moe.num_experts,
            ).to(self.context.runtime.device)
        elif self.config.parallelism.tp > 1:
            model = TensorParallelTransformerLM(
                self.config.model,
                self.context.groups,
            ).to(self.context.runtime.device)
        else:
            model = TransformerLM(self.config.model).to(self.context.runtime.device)

        model = maybe_cast_model(model, self.config.precision)
        model = apply_activation_checkpointing(
            model,
            self.config.activation_checkpointing,
        )
        return model

    def _build_moe_pipeline_stage(self) -> nn.Module:
        assert self.context is not None

        groups = self.context.groups
        stage_index = groups.pp_ranks.index(groups.rank)
        num_stages = len(groups.pp_ranks)

        from meshtrain.core.distributed.placement import split_range

        layer_range = split_range(
            size=self.config.model.n_layers,
            parts=num_stages,
            index=stage_index,
            require_even=False,
        )

        return MoETransformerPipelineStage(
            self.config.model,
            groups,
            layer_start=layer_range.start,
            layer_end=layer_range.end,
            num_experts=self.config.moe.num_experts,
            is_first=stage_index == 0,
            is_last=stage_index == num_stages - 1,
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer_config = self.config.optimizer

        if optimizer_config.name != "adamw":
            raise ValueError(f"unsupported optimizer: {optimizer_config.name}")

        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )

    def _build_dataloader(self) -> DataLoader:
        assert self.context is not None

        train_path = Path(self.config.data.train_path)
        if not train_path.exists():
            raise FileNotFoundError(
                f"train shard not found: {train_path}. Run "
                "`python scripts/download_parameter_golf.py --train-shards 1` "
                "or pass a valid DataConfig.train_path."
            )

        dataset = TokenShardDataset.from_files([train_path])
        packer = CausalLMPacker(dataset, seq_len=self.config.data.seq_len)
        sampler = DPSampler(
            num_examples=len(packer),
            global_batch_size=self.config.training.global_batch_size,
            dp_rank=self.context.placement.coord.dp,
            dp_size=self.config.parallelism.dp,
        )

        return DataLoader(
            packer=packer,
            sampler=sampler,
            device=self.context.runtime.device,
        )

    def _target_ids_for_loss(self, target_ids: torch.Tensor) -> torch.Tensor:
        assert self.context is not None

        if self.config.parallelism.cp == 1:
            return target_ids

        return shard_sequence(
            target_ids,
            self.context.groups,
            seq_dim=1,
        )

    def train_step(self) -> TrainMetrics:
        if self.uses_pipeline_parallel:
            return self._pipeline_gpipe_train_step()

        return self._single_stage_train_step()

    def _single_stage_train_step(self) -> TrainMetrics:
        if self.context is None:
            self.setup()

        assert self.context is not None
        assert self.model is not None
        assert self.optimizer is not None
        assert self.dataloader is not None

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._debug_step("single:zero_grad_done")

        batch = self.dataloader.get_batch(self.state.global_batch_id)
        self._debug_step("single:batch_done")

        with autocast_context(
            self.config.precision,
            device=self.context.runtime.device,
        ):
            self._debug_step("single:forward_start")
            logits = self.model(batch.input_ids)
            self._debug_step("single:forward_done")
            loss_out = next_token_loss(logits, self._target_ids_for_loss(batch.target_ids))
            self._debug_step("single:loss_done")

        self._debug_step("single:loss_stats_sync_start")
        global_stats = sync_loss_stats(
            loss_out.loss_sum,
            loss_out.num_tokens,
            self.context.groups,
        )
        self._debug_step("single:loss_stats_sync_done")
        backward_loss = loss_out.loss_sum / global_stats.num_tokens.clamp_min(1)

        if self.scaler is not None:
            self._debug_step("single:backward_start")
            self.scaler.scale(backward_loss).backward()
            self._debug_step("single:backward_done")
            self.scaler.unscale_(self.optimizer)
        else:
            self._debug_step("single:backward_start")
            backward_loss.backward()
            self._debug_step("single:backward_done")

        self._debug_step("single:grad_sync_start")
        sync_gradients(self.model, self.context.groups)
        self._debug_step("single:grad_sync_done")

        if self.scaler is not None:
            self._debug_step("single:optimizer_start")
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self._debug_step("single:optimizer_start")
            self.optimizer.step()
        self._debug_step("single:optimizer_done")

        metrics = TrainMetrics(
            step=self.state.step,
            global_batch_id=self.state.global_batch_id,
            loss=float(global_stats.loss.detach().cpu()),
            loss_sum=float(global_stats.loss_sum.detach().cpu()),
            num_tokens=int(global_stats.num_tokens.detach().cpu()),
        )
        self._advance_state()
        return metrics

    def _pipeline_gpipe_train_step(self) -> TrainMetrics:
        if self.context is None:
            self.setup()

        assert self.context is not None
        assert self.pipeline_stage is not None
        assert self.optimizer is not None
        assert self.dataloader is not None

        stage = self.pipeline_stage
        stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._debug_step("pipeline:zero_grad_done")

        batch = self.dataloader.get_batch(self.state.global_batch_id)
        self._debug_step("pipeline:batch_done")
        num_microbatches = self._num_microbatches()
        input_microbatches = list(batch.input_ids.chunk(num_microbatches, dim=0))
        target_microbatches = list(batch.target_ids.chunk(num_microbatches, dim=0))

        microbatch_size = input_microbatches[0].shape[0]
        activation_seq_len = (
            self.config.model.seq_len // self.config.parallelism.cp
            if self.config.parallelism.cp > 1
            else self.config.model.seq_len
        )
        activation_shape = (
            microbatch_size,
            activation_seq_len,
            self.config.model.dim,
        )

        def loss_fn(logits: torch.Tensor, microbatch_id: int) -> torch.Tensor:
            loss_out = next_token_loss(
                logits,
                self._target_ids_for_loss(target_microbatches[microbatch_id]),
            )
            return loss_out.loss / num_microbatches

        with autocast_context(
            self.config.precision,
            device=self.context.runtime.device,
        ):
            self._debug_step("pipeline:gpipe_start")
            losses = gpipe_forward_backward(
                stage,
                input_microbatches if stage.is_first else None,
                num_microbatches=num_microbatches,
                activation_shape=activation_shape,
                dtype=self._module_dtype(stage),
                device=self.context.runtime.device,
                loss_fn=loss_fn if stage.is_last else None,
            )
            self._debug_step("pipeline:gpipe_done")

        self._debug_step("pipeline:grad_sync_start")
        sync_gradients(stage, self.context.groups)
        self._debug_step("pipeline:grad_sync_done")
        self._debug_step("pipeline:optimizer_start")
        self.optimizer.step()
        self._debug_step("pipeline:optimizer_done")

        loss_tensor = torch.zeros((), device=self.context.runtime.device)
        token_tensor = torch.zeros((), device=self.context.runtime.device, dtype=torch.long)
        if stage.is_last:
            assert losses is not None
            loss_tensor = torch.stack(losses).sum().to(dtype=torch.float32)
            token_tensor = torch.tensor(
                batch.num_tokens,
                device=self.context.runtime.device,
                dtype=torch.long,
            )

        if self.context.runtime.is_distributed:
            self._debug_step("pipeline:loss_reduce_start")
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            self._debug_step("pipeline:loss_reduce_done")

        num_tokens = int(token_tensor.detach().cpu())
        loss = float(loss_tensor.detach().cpu()) / self.config.parallelism.dp
        loss_sum = loss * num_tokens

        metrics = TrainMetrics(
            step=self.state.step,
            global_batch_id=self.state.global_batch_id,
            loss=loss,
            loss_sum=loss_sum,
            num_tokens=num_tokens,
        )
        self._advance_state()
        return metrics

    def _num_microbatches(self) -> int:
        training = self.config.training
        if training.global_batch_size % training.microbatch_size != 0:
            raise ValueError("global_batch_size must be divisible by microbatch_size")

        return training.global_batch_size // training.microbatch_size

    def _module_dtype(self, module: nn.Module) -> torch.dtype:
        for parameter in module.parameters():
            return parameter.dtype
        return torch.float32

    def _advance_state(self) -> None:
        self.state = TrainerState(
            step=self.state.step + 1,
            global_batch_id=self.state.global_batch_id + 1,
        )

    def train(self) -> None:
        if self.context is None:
            self.setup()

        assert self.context is not None
        assert self.optimizer is not None

        while self.state.step < self.config.training.max_steps:
            metrics = self.train_step()
            next_step = metrics.step + 1

            if self.is_rank_zero and next_step % self.config.training.log_every == 0:
                print(
                    f"step={next_step} "
                    f"global_batch_id={metrics.global_batch_id} "
                    f"loss={metrics.loss:.6f} "
                    f"num_tokens={metrics.num_tokens}"
                )

            if (
                not self.uses_pipeline_parallel
                and next_step % self.config.training.checkpoint_every == 0
            ):
                save_checkpoint(
                    model=self._train_module(),
                    optimizer=self.optimizer,
                    config=self.config,
                    step=next_step,
                    checkpoint_config=self.config.checkpoint,
                    rank=self.context.runtime.rank,
                )

    def close(self) -> None:
        if self.context is not None:
            shutdown_engine_context(self.context)
            self.context = None

    def __enter__(self) -> EngineTrainer:
        self.setup()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
