from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .batch import Batch, collate_examples
from .dataset import TokenShardDataset
from .packing import CausalLMPacker
from .sampler import DPSampler


@dataclass(frozen=True)
class DataLoader:
    packer: CausalLMPacker
    sampler: DPSampler
    device: torch.device | str | None = None

    def get_batch(self, global_batch_id: int) -> Batch:
        plan = self.sampler.plan_batch(global_batch_id)
        examples = [
            self.packer.get_example(sample_id)
            for sample_id in plan.local_sample_ids
        ]

        return collate_examples(
            examples,
            global_batch_id=plan.global_batch_id,
            local_batch_range=plan.local_batch_range,
            device=self.device,
        )


if __name__ == "__main__":
    train_path = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    if not train_path.exists():
        raise FileNotFoundError(
            "No train shard found. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first."
        )

    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=16)

    for dp_rank in range(2):
        sampler = DPSampler(
            num_examples=len(packer),
            global_batch_size=8,
            dp_rank=dp_rank,
            dp_size=2,
        )
        loader = DataLoader(packer=packer, sampler=sampler)
        batch = loader.get_batch(0)

        print(f"dp_rank={dp_rank}")
        print("  input_shape:", tuple(batch.input_ids.shape))
        print("  target_shape:", tuple(batch.target_ids.shape))
        print("  sample_ids:", batch.sample_ids.tolist())
        print("  local_batch_range:", batch.local_batch_range)
        print("  global_batch_id:", batch.global_batch_id)
        print("  num_tokens:", batch.num_tokens)
