from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.model.standard_transformer import TransformerConfig
from meshtrain.parallelism.context_parallelism import (
    context_parallel_range,
    ring_causal_attention,
    shard_sequence,
)
from meshtrain.parallelism.expert_parallelism import (
    combine_expert_outputs,
    dispatch_tokens_to_experts,
    expert_parallel_range,
    run_local_experts,
)
from meshtrain.parallelism.tensor_parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def _tp_size(groups: ParallelGroups) -> int:
    return len(groups.tp_ranks)


def _tp_is_active(groups: ParallelGroups) -> bool:
    return groups.tp_group is not None and len(groups.tp_ranks) > 1


def _cp_is_active(groups: ParallelGroups) -> bool:
    return groups.cp_group is not None and len(groups.cp_ranks) > 1


def _ep_is_active(groups: ParallelGroups) -> bool:
    return groups.ep_group is not None and len(groups.ep_ranks) > 1


class CPEnabledSelfAttention(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
    ):
        super().__init__()

        if config.dim % config.n_heads != 0:
            raise ValueError("model dim must be divisible by n_heads")
        if config.n_heads % _tp_size(groups) != 0:
            raise ValueError("n_heads must be divisible by tp")

        self.config = config
        self.groups = groups
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.local_heads = config.n_heads // _tp_size(groups)
        self.local_dim = self.local_heads * self.head_dim

        if _tp_is_active(groups):
            self.qkv = ColumnParallelLinear(
                config.dim,
                3 * config.dim,
                groups,
                bias=False,
                gather_output=False,
            )
            self.out_proj = RowParallelLinear(
                config.dim,
                config.dim,
                groups,
                bias=False,
                input_is_parallel=True,
            )
        else:
            self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
            self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"input dim {dim} does not match attention dim {self.dim}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.local_heads, self.head_dim).transpose(1, 2)

        if _cp_is_active(self.groups):
            attn = ring_causal_attention(
                q,
                k,
                v,
                self.groups,
                sequence_length=self.config.seq_len,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )

        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.local_dim)
        return self.out_proj(attn)


class LocalExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertParallelMLP(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
        *,
        num_experts: int,
    ):
        super().__init__()

        self.config = config
        self.groups = groups
        self.num_experts = num_experts
        self.shard = expert_parallel_range(num_experts, groups)

        self.router = nn.Linear(config.dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                LocalExpertMLP(config.dim, config.mlp_hidden_dim)
                for _ in range(self.shard.num_local_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])

        router_logits = self.router(tokens)
        expert_ids = router_logits.argmax(dim=-1).to(torch.long)

        dispatch = dispatch_tokens_to_experts(
            tokens,
            expert_ids,
            num_experts=self.num_experts,
            groups=self.groups,
        )
        expert_outputs = run_local_experts(
            dispatch.received_tokens,
            dispatch.received_expert_ids,
            self.experts,
            self.shard,
        )
        combined = combine_expert_outputs(
            expert_outputs,
            dispatch,
            original_num_tokens=tokens.shape[0],
            groups=self.groups,
        )
        if _tp_is_active(self.groups) and _ep_is_active(self.groups):
            dist.barrier(group=self.groups.tp_group)
        return combined.view(original_shape)


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
        *,
        num_experts: int,
    ):
        super().__init__()

        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = CPEnabledSelfAttention(config, groups)
        self.mlp_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = ExpertParallelMLP(
            config,
            groups,
            num_experts=num_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MoETransformerLM(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
        *,
        num_experts: int,
    ):
        super().__init__()

        self.config = config
        self.groups = groups
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.position_emb = nn.Embedding(config.seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(config, groups, num_experts=num_experts)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len != self.config.seq_len:
            raise ValueError(
                f"MoETransformerLM currently expects fixed seq_len={self.config.seq_len}, "
                f"got {seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.position_emb(positions)[None, :, :]

        if _cp_is_active(self.groups):
            x = shard_sequence(x, self.groups, seq_dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)


class MoETransformerPipelineStage(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        groups: ParallelGroups,
        *,
        layer_start: int,
        layer_end: int,
        num_experts: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()

        self.config = config
        self.groups = groups
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = is_first
        self.is_last = is_last

        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(config, groups, num_experts=num_experts)
                for _ in range(layer_end - layer_start)
            ]
        )
        self.token_emb = nn.Embedding(config.vocab_size, config.dim) if is_first else None
        self.position_emb = nn.Embedding(config.seq_len, config.dim) if is_first else None
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps) if is_last else None
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False) if is_last else None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def prev_rank(self) -> int | None:
        from meshtrain.parallelism.pipeline_parallel.p2p import pipeline_prev_rank

        return pipeline_prev_rank(self.groups)

    @property
    def next_rank(self) -> int | None:
        from meshtrain.parallelism.pipeline_parallel.p2p import pipeline_next_rank

        return pipeline_next_rank(self.groups)

    def forward_local(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            if x.ndim != 2:
                raise ValueError("first MoE pipeline stage expects input_ids [B, T]")

            batch, seq_len = x.shape
            if seq_len != self.config.seq_len:
                raise ValueError(
                    f"MoE pipeline stage expects seq_len={self.config.seq_len}, got {seq_len}"
                )

            assert self.token_emb is not None
            assert self.position_emb is not None
            positions = torch.arange(seq_len, device=x.device)
            x = self.token_emb(x) + self.position_emb(positions)[None, :, :]

            if _cp_is_active(self.groups):
                x = shard_sequence(x, self.groups, seq_dim=1)

        for layer in self.layers:
            x = layer(x)

        if self.is_last:
            assert self.norm is not None
            assert self.lm_head is not None
            x = self.norm(x)
            x = self.lm_head(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_local(x)
