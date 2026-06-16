from __future__ import annotations 
from dataclasses import dataclass  
from pathlib import Path  

import torch  
import torch.nn as nn  

from meshtrain.data.dataset import TokenShardDataset
from meshtrain.data.packing import CausalLMPacker
from meshtrain.data.sampler import DPSampler
from meshtrain.data.dataloader import DataLoader
from meshtrain.model.standard_transformer import TransformerConfig, TransformerLM
from meshtrain.training.loss import next_token_loss, LossOutput  



@dataclass(frozen=True)  
class ReferenceTrainConfig:  
    seq_len: int = 64
    global_batch_size: int = 8  

    vocab_size: int = 1024  
    dim: int = 128  
    n_layers: int = 2  
    n_heads: int = 4
    mlp_hidden_dim: int = 256

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    device: str = "cuda"if torch.cuda.is_available() else "cpu"  


def build_optimizer(model, config): 
    decay_params = []  
    no_decay_params = [] 

    for name, param in model.named_parameters():  
        if not param.requires_grad:  
            continue  
        
        if param.ndim >= 2:  
            decay_params.append(param)
        
        else: 
            no_decay_params.append(param)
    
    return torch.optim.AdamW(
         [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


@dataclass(frozen=True)  
class TrainStepOutput:  
    step: int 
    loss: float 
    loss_sum: float 
    num_tokens: int  
    grad_norm: float 


def grad_norm(model):
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5  


def train_step(model, optimizer, batch, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(batch.input_ids)
    loss_output = next_token_loss(logits, batch.target_ids)

    loss_output.loss.backward()

    norm = grad_norm(model)

    optimizer.step()

    return TrainStepOutput(
        step=step,
        loss=float(loss_output.loss.detach()),
        loss_sum=float(loss_output.loss_sum.detach()),
        num_tokens=int(loss_output.num_tokens),
        grad_norm=norm,
    ) 


def build_reference_dataloader(config):
    train_path = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=config.seq_len)

    sampler = DPSampler(
        num_examples=len(packer),
        global_batch_size=config.global_batch_size,
        dp_rank=0,
        dp_size=1,
    )

    return DataLoader(
        packer=packer,
        sampler=sampler,
        device=config.device,
    )  



if __name__ == "__main__":
    torch.manual_seed(0)

    config = ReferenceTrainConfig()

    dataloader = build_reference_dataloader(config)

    model_config = TransformerConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
    )

    model = TransformerLM(model_config).to(config.device)
    optimizer = build_optimizer(model, config)

    for step in range(10000):
        batch = dataloader.get_batch(step)
        out = train_step(model, optimizer, batch, step)
        print(out)   
