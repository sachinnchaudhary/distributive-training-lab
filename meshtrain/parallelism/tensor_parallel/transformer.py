from __future__ import annotations

import os

import torch
import torch.nn as nn

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.model.standard_transformer import TransformerConfig, TransformerLM
from meshtrain.parallelism.tensor_parallel.attention import TensorParallelSelfAttention
from meshtrain.parallelism.tensor_parallel.mlp import TensorParallelMLP


def _debug_tp(groups: ParallelGroups, message: str) -> None:
    if os.environ.get("MESHTRAIN_TP_DEBUG", "0") != "1":
        return

    if os.environ.get("MESHTRAIN_TP_SYNC_DEBUG", "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"rank={groups.rank} tp_group={groups.tp_ranks} tp_transformer:{message}",
        flush=True,
    )


class TensorParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
    ):
        super().__init__()

        self.groups = groups
        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = TensorParallelSelfAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            groups=groups,
            dropout=config.dropout,
            bias=False,
        )
        self.mlp_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = TensorParallelMLP(
            dim=config.dim,
            hidden_dim=config.mlp_hidden_dim,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _debug_tp(self.groups, "block:attn_norm_start")
        attn_input = self.attn_norm(x)
        _debug_tp(self.groups, "block:attn_norm_done")

        _debug_tp(self.groups, "block:attn_start")
        attn_output = self.attn(attn_input)
        _debug_tp(self.groups, "block:attn_done")

        _debug_tp(self.groups, "block:attn_residual_start")
        x = x + attn_output
        _debug_tp(self.groups, "block:attn_residual_done")

        _debug_tp(self.groups, "block:mlp_norm_start")
        mlp_input = self.mlp_norm(x)
        _debug_tp(self.groups, "block:mlp_norm_done")

        _debug_tp(self.groups, "block:mlp_start")
        mlp_output = self.mlp(mlp_input)
        _debug_tp(self.groups, "block:mlp_done")

        _debug_tp(self.groups, "block:mlp_residual_start")
        x = x + mlp_output
        _debug_tp(self.groups, "block:mlp_residual_done")
        return x

    @torch.no_grad()
    def load_from_block(self, block: nn.Module) -> None:
        self.attn_norm.load_state_dict(block.attn_norm.state_dict())
        self.attn.load_from_attention(block.attn)
        self.mlp_norm.load_state_dict(block.mlp_norm.state_dict())
        self.mlp.load_from_mlp(block.mlp)


class TensorParallelTransformerLM(nn.Module):
    """
    Tensor-parallel TransformerLM with replicated embeddings and LM head.

    The expensive attention and MLP projections are tensor-parallel. Embeddings,
    final norm, and lm_head stay replicated in this first integration so the
    trainer can wire TP before adding vocab-parallel output projection.
    """

    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
    ):
        super().__init__()

        self.config = config
        self.groups = groups
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.position_emb = nn.Embedding(config.seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [
                TensorParallelTransformerBlock(config, groups)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _debug_tp(self.groups, f"lm:forward_start shape={tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"input seq_len {seq_len} exceeds model seq_len {self.config.seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        _debug_tp(self.groups, "lm:embedding_start")
        x = self.token_emb(input_ids) + self.position_emb(positions)[None, :, :]
        _debug_tp(self.groups, f"lm:embedding_done shape={tuple(x.shape)}")

        for block_index, block in enumerate(self.blocks):
            _debug_tp(self.groups, f"lm:block_{block_index}_start")
            x = block(x)
            _debug_tp(self.groups, f"lm:block_{block_index}_done")

        _debug_tp(self.groups, "lm:head_start")
        x = self.norm(x)
        output = self.lm_head(x)
        _debug_tp(self.groups, f"lm:head_done shape={tuple(output.shape)}")
        return output

    @torch.no_grad()
    def load_from_transformer_lm(self, model: TransformerLM) -> None:
        self.token_emb.load_state_dict(model.token_emb.state_dict())
        self.position_emb.load_state_dict(model.position_emb.state_dict())

        for tp_block, source_block in zip(self.blocks, model.blocks, strict=True):
            tp_block.load_from_block(source_block)

        self.norm.load_state_dict(model.norm.state_dict())
        self.lm_head.load_state_dict(model.lm_head.state_dict())
