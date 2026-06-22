from __future__ import annotations

from dataclasses import dataclass

from meshtrain.core.distributed.groups import ParallelGroups, build_parallel_groups
from meshtrain.core.distributed.mesh import ParallelDims, RankMesh
from meshtrain.core.distributed.placement import RankPlacement, placement_for_rank
from meshtrain.core.distributed.runtime import (
    RuntimeContext,
    init_runtime,
    shutdown_runtime,
)
from meshtrain.engine.config import EngineConfig


@dataclass(frozen=True)
class EngineContext:
    config: EngineConfig
    runtime: RuntimeContext
    dims: ParallelDims
    mesh: RankMesh
    groups: ParallelGroups
    placement: RankPlacement


def build_parallel_dims(config: EngineConfig) -> ParallelDims:
    parallelism = config.parallelism

    return ParallelDims(
        dp=parallelism.dp,
        pp=parallelism.pp,
        tp=parallelism.tp,
        cp=parallelism.cp,
        ep=parallelism.ep,
    )


def build_rank_placement(
    config: EngineConfig,
    mesh: RankMesh,
    rank: int,
) -> RankPlacement:
    return placement_for_rank(
        rank,
        mesh,
        num_layers=config.model.n_layers,
        global_batch_size=config.training.global_batch_size,
        sequence_length=config.model.seq_len,
        tensor_shape=(config.model.dim, config.model.mlp_hidden_dim),
        tensor_dim=1,
        num_experts=None,
    )


def build_engine_context(config: EngineConfig) -> EngineContext:
    runtime = init_runtime()

    try:
        config.validate(world_size=runtime.world_size)

        dims = build_parallel_dims(config)
        mesh = RankMesh(dims, world_size=runtime.world_size)
        groups = build_parallel_groups(runtime, mesh)
        placement = build_rank_placement(config, mesh, runtime.rank)

        return EngineContext(
            config=config,
            runtime=runtime,
            dims=dims,
            mesh=mesh,
            groups=groups,
            placement=placement,
        )
    except Exception:
        shutdown_runtime()
        raise


def shutdown_engine_context(context: EngineContext | None = None) -> None:
    shutdown_runtime()
