from __future__ import annotations 
from dataclasses import dataclass  
import numpy as np  

from .dataset import TokenShardDataset

@dataclass(frozen=True)  
class PackedExample:  
    sample_id: int  
    token_start: int  
    input_ids: np.ndarray
    target_ids: np.ndarray


class CausalLMPacker:  
    def __init__(self, dataset: TokenShardDataset, seq_len: int, stride: int | None= None):  
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")

        self.dataset = dataset
        self.seq_len = seq_len
        self.stride = stride or seq_len

        if self.stride < 1:
            raise ValueError(f"stride must be at least 1, got {self.stride}")

        if len(dataset) < self.seq_len + 1:
            raise ValueError(
                f"dataset has {len(dataset)} tokens, but seq_len={self.seq_len} needs at least {self.seq_len + 1}"
            )

        self.num_examples = (len(dataset) - (self.seq_len + 1)) // self.stride + 1

    def __len__(self) -> int:  

        return self.num_examples 
    
    def get_example(self, sample_id: int) -> PackedExample:  
        if sample_id < 0 or sample_id >= self.num_examples:  
            raise IndexError(f"sample_id must be in [0, {self.num_examples}), got {sample_id}")   

        token_start = sample_id * self.stride  
        tokens = self.dataset.get_tokens(
            token_start, 
            token_start + self.seq_len + 1, 
        )

        input_ids = tokens[:-1]  
        target_ids = tokens[1:]  

        return PackedExample( 
            sample_id=sample_id, 
            token_start=token_start, 
            input_ids=input_ids, 
            target_ids=target_ids,
        )       


if __name__ == "__main__":
    from pathlib import Path

    from meshtrain.data.tokenizer import load_tokenizer

    train_path = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    if not train_path.exists():
        raise FileNotFoundError(
            "No train shard found. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first."
        )

    dataset = TokenShardDataset.from_files([train_path])
    packer = CausalLMPacker(dataset, seq_len=16)
    example = packer.get_example(0)

    print("num_examples:", len(packer))
    print("sample_id:", example.sample_id)
    print("token_start:", example.token_start)
    print("input_ids:", example.input_ids.tolist())
    print("target_ids:", example.target_ids.tolist())
    print("shift_ok:", bool(np.array_equal(example.input_ids[1:], example.target_ids[:-1])))

    tokenizer = load_tokenizer()
    print("decoded_input:", tokenizer.decode(example.input_ids.tolist()))
