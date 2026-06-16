from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meshtrain.data.dataloader import DataLoader
from meshtrain.data.dataset import TokenShardDataset
from meshtrain.data.packing import CausalLMPacker
from meshtrain.data.sampler import DPSampler
from meshtrain.model.standard_transformer import TransformerConfig, TransformerLM
from meshtrain.training.loss import LossOutput, next_token_loss
from meshtrain.training.reference_training import ReferenceTrainConfig, build_optimizer


@dataclass(frozen=True)
class DPCorrectnessConfig:
    seed: int = 0
    seq_len: int = 32
    global_batch_size: int = 8
    dp_size: int = 2

    vocab_size: int = 1024
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_hidden_dim: int = 256

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    atol: float = 1e-5
    rtol: float = 1e-4
    train_path: str = "data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"


def build_loader(config: DPCorrectnessConfig, dp_rank: int, dp_size: int) -> DataLoader:
    train_path = Path(config.train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            "No train shard found. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first."
        )

    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=config.seq_len)
    sampler = DPSampler(
        num_examples=len(packer),
        global_batch_size=config.global_batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    return DataLoader(packer=packer, sampler=sampler, device=config.device)


def build_model(config: DPCorrectnessConfig) -> TransformerLM:
    model_config = TransformerConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
    )
    return TransformerLM(model_config).to(config.device)


def build_train_config(config: DPCorrectnessConfig) -> ReferenceTrainConfig:
    return ReferenceTrainConfig(
        seq_len=config.seq_len,
        global_batch_size=config.global_batch_size,
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
        device=config.device,
    )


def reference_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batch,
) -> LossOutput:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(batch.input_ids)
    loss_out = next_token_loss(logits, batch.target_ids)
    loss_out.loss.backward()
    optimizer.step()
    return loss_out


def simulated_dp_forward_backward(
    models: list[TransformerLM],
    optimizers: list[torch.optim.Optimizer],
    batches,
) -> tuple[torch.Tensor, torch.Tensor]:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    local_losses: list[LossOutput] = []
    global_num_tokens = torch.zeros((), device=batches[0].input_ids.device, dtype=torch.long)

    for model, batch in zip(models, batches):
        logits = model(batch.input_ids)
        loss_out = next_token_loss(logits, batch.target_ids)
        local_losses.append(loss_out)
        global_num_tokens = global_num_tokens + loss_out.num_tokens

    global_loss_sum = sum(loss_out.loss_sum for loss_out in local_losses)

    for loss_out in local_losses:
        scaled_local_loss = loss_out.loss_sum / global_num_tokens.clamp_min(1)
        scaled_local_loss.backward()

    sync_simulated_dp_grads(models)

    for optimizer in optimizers:
        optimizer.step()

    return global_loss_sum.detach(), global_num_tokens.detach()


def sync_simulated_dp_grads(models: list[TransformerLM]) -> None:
    named_params = [dict(model.named_parameters()) for model in models]
    names = list(named_params[0].keys())

    for name in names:
        grads = [params[name].grad for params in named_params]
        if all(grad is None for grad in grads):
            continue
        if any(grad is None for grad in grads):
            raise RuntimeError(f"missing gradient for parameter {name}")

        grad_sum = torch.stack([grad.detach() for grad in grads]).sum(dim=0)
        for params in named_params:
            params[name].grad.copy_(grad_sum)


def max_param_diff(model_a: TransformerLM, model_b: TransformerLM) -> float:
    max_diff = 0.0
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    if params_a.keys() != params_b.keys():
        raise RuntimeError("model parameter sets do not match")

    for name in params_a:
        diff = (params_a[name].detach() - params_b[name].detach()).abs().max().item()
        max_diff = max(max_diff, diff)

    return max_diff


def main() -> None:
    config = DPCorrectnessConfig()
    torch.manual_seed(config.seed)

    reference_model = build_model(config)
    initial_state = copy.deepcopy(reference_model.state_dict())

    dp_models = [build_model(config) for _ in range(config.dp_size)]
    reference_model.load_state_dict(initial_state)
    for model in dp_models:
        model.load_state_dict(initial_state)

    train_config = build_train_config(config)
    reference_optimizer = build_optimizer(reference_model, train_config)
    dp_optimizers = [build_optimizer(model, train_config) for model in dp_models]

    reference_loader = build_loader(config, dp_rank=0, dp_size=1)
    dp_loaders = [
        build_loader(config, dp_rank=rank, dp_size=config.dp_size)
        for rank in range(config.dp_size)
    ]

    reference_batch = reference_loader.get_batch(0)
    dp_batches = [loader.get_batch(0) for loader in dp_loaders]

    print("reference_sample_ids:", reference_batch.sample_ids.tolist())
    for rank, batch in enumerate(dp_batches):
        print(f"dp_rank={rank} sample_ids:", batch.sample_ids.tolist())

    reference_loss = reference_step(reference_model, reference_optimizer, reference_batch)
    dp_loss_sum, dp_num_tokens = simulated_dp_forward_backward(
        dp_models,
        dp_optimizers,
        dp_batches,
    )
    dp_loss = dp_loss_sum / dp_num_tokens.clamp_min(1)

    print("reference_loss:", float(reference_loss.loss.detach()))
    print("dp_global_loss:", float(dp_loss.detach()))

    passed = True
    for rank, model in enumerate(dp_models):
        diff = max_param_diff(reference_model, model)
        rank_passed = diff <= config.atol
        passed = passed and rank_passed
        print(f"max_param_diff_rank{rank}:", diff)

    print("passed:", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
