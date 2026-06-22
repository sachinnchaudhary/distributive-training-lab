from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from meshtrain.engine.config import ActivationCheckpointConfig


class CheckpointedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if not self.training:
            return self.module(*args, **kwargs)

        if kwargs:
            return checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                **kwargs,
            )

        return checkpoint(
            self.module,
            *args,
            use_reentrant=False,
        )


def _should_checkpoint_block(
    block_index: int,
    config: ActivationCheckpointConfig,
) -> bool:
    return block_index % config.every_n_layers == 0


def apply_activation_checkpointing(
    model: nn.Module,
    config: ActivationCheckpointConfig,
) -> nn.Module:
    if not config.enabled:
        return model

    if config.granularity != "block":
        raise ValueError(f"unsupported activation checkpoint granularity: {config.granularity}")

    blocks = getattr(model, "blocks", None)
    if blocks is None or not isinstance(blocks, nn.ModuleList):
        raise ValueError("block activation checkpointing expects model.blocks as nn.ModuleList")

    for block_index, block in enumerate(blocks):
        if _should_checkpoint_block(block_index, config):
            blocks[block_index] = CheckpointedModule(block)

    return model


def count_checkpointed_modules(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, CheckpointedModule)
    )
