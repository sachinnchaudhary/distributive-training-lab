from __future__ import annotations  
from dataclasses import dataclass  


import torch  
import torch.nn.functional as F  


@dataclass(frozen=True)  
class LossOutput:  
    loss: torch.Tensor 
    loss_sum: torch.Tensor  
    num_tokens: torch.Tensor  


def next_token_loss(
        logits: torch.Tensor, 
        target_ids: torch.Tensor,  
        *, 
        ignore_index: int = -100,  
) -> LossOutput:  
   
   
   if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
   if target_ids.ndim != 2:
        raise ValueError(
            f"target_ids must have shape [B, T], got {tuple(target_ids.shape)}"
        )
   if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            f"logits batch/seq shape {tuple(logits.shape[:2])} must match "
            f"target_ids shape {tuple(target_ids.shape)}"
        )
   
   
   vocab_size = logits.shape[-1]  

   loss_sum = F.cross_entropy(
       logits.reshape(-1, vocab_size), 
       target_ids.reshape(-1), 
       ignore_index=ignore_index, 
       reduction="sum",
   )

   num_tokens = (target_ids != ignore_index).sum()  
   loss = loss_sum / num_tokens.clamp_min(1)  

   return LossOutput(
       loss=loss, 
       loss_sum=loss_sum, 
       num_tokens=num_tokens,
   )  


if __name__ == "__main__":
    from pathlib import Path

    from meshtrain.data.dataloader import DataLoader
    from meshtrain.data.dataset import TokenShardDataset
    from meshtrain.data.packing import CausalLMPacker
    from meshtrain.data.sampler import DPSampler
    from meshtrain.model.standard_transformer import TransformerConfig, TransformerLM

    train_path = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    if not train_path.exists():
        raise FileNotFoundError(
            "No train shard found. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first."
        )

    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=16)
    sampler = DPSampler(
        num_examples=len(packer),
        global_batch_size=4,
        dp_rank=0,
        dp_size=1,
    )
    loader = DataLoader(packer=packer, sampler=sampler)
    batch = loader.get_batch(0)

    config = TransformerConfig(
        vocab_size=1024,
        seq_len=16,
        dim=128,
        n_layers=2,
        n_heads=4,
        mlp_hidden_dim=256,
    )
    model = TransformerLM(config)
    logits = model(batch.input_ids)
    loss_output = next_token_loss(logits, batch.target_ids)

    print("logits_shape:", tuple(logits.shape))
    print("loss:", float(loss_output.loss.detach()))
    print("loss_sum:", float(loss_output.loss_sum.detach()))
    print("num_tokens:", int(loss_output.num_tokens))

