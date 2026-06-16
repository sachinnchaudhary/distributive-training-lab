from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import numpy as np

from .dataset import TokenShardDataset
from .packing import CausalLMPacker, PackedExample


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    sample_ids: torch.Tensor
    global_batch_id: int
    local_batch_range: tuple[int, int]
    num_tokens: int

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]


def collate_examples(
    examples: Sequence[PackedExample],
    *,
    global_batch_id: int,
    local_batch_range: tuple[int, int],
    device: torch.device | str | None = None,
) -> Batch:
    if not examples:
        raise ValueError("cannot collate an empty example list")

    input_array = np.stack([example.input_ids for example in examples])
    target_array = np.stack([example.target_ids for example in examples])

    input_ids = torch.as_tensor(input_array, dtype=torch.long, device=device)
    target_ids = torch.as_tensor(target_array, dtype=torch.long, device=device)
    sample_ids = torch.tensor(
        [example.sample_id for example in examples],
        dtype=torch.long,
        device=device,
    )

    return Batch(
        input_ids=input_ids,
        target_ids=target_ids,
        sample_ids=sample_ids,
        global_batch_id=global_batch_id,
        local_batch_range=local_batch_range,
        num_tokens=target_ids.numel(),
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
    examples = [packer.get_example(i) for i in range(4)]
    batch = collate_examples(
        examples,
        global_batch_id=0,
        local_batch_range=(0, 4),
    )

    print("input_shape:", tuple(batch.input_ids.shape))
    print("target_shape:", tuple(batch.target_ids.shape))
    print("sample_ids:", batch.sample_ids.tolist())
    print("global_batch_id:", batch.global_batch_id)
    print("local_batch_range:", batch.local_batch_range)
    print("num_tokens:", batch.num_tokens)
