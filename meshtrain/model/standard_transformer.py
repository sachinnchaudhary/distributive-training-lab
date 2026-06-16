from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 1024
    seq_len: int = 256
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_hidden_dim: int = 1024
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(attn)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.position_emb = nn.Embedding(config.seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
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
        batch, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"input seq_len {seq_len} exceeds model seq_len {self.config.seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.position_emb(positions)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=1024,
        seq_len=16,
        dim=128,
        n_layers=2,
        n_heads=4,
        mlp_hidden_dim=256,
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    logits = model(input_ids)

    print("input_shape:", tuple(input_ids.shape))
    print("logits_shape:", tuple(logits.shape))
    print("parameters:", count_parameters(model))
