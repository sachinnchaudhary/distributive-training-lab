from __future__ import annotations  

import bisect
from dataclasses import dataclass  
from pathlib import Path
from typing import Sequence  

import numpy as np    


PARAMETER_GOLF_HEADER_BYTES = 1024


@dataclass(frozen=True)  
class TokenShard:  
    path: Path  
    num_tokens: int  
    data_offset_bytes: int = PARAMETER_GOLF_HEADER_BYTES

class TokenShardDataset:  
    def __init__(self, shards: Sequence[TokenShard], dtype=np.uint16):  
        self.shards = list(shards)  
        self.dtype = np.dtype(dtype)  

        running = 0 
        self.cumulative_lengths = []
        for shard in self.shards:  
            running += shard.num_tokens
            self.cumulative_lengths.append(running)
        self.total_tokens = running
        self.arrays = [
            np.memmap(
                shard.path,
                dtype=self.dtype,
                mode="r",
                offset=shard.data_offset_bytes,
                shape=(shard.num_tokens,),
            )
            for shard in self.shards
        ]

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        dtype=np.uint16,
        header_bytes: int = PARAMETER_GOLF_HEADER_BYTES,
    ):
        shards = []

        for path in paths:
            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"token shard does not exist: {path}")

            num_bytes = path.stat().st_size
            itemsize = np.dtype(dtype).itemsize
            if header_bytes < 0:
                raise ValueError(f"header_bytes must be non-negative, got {header_bytes}")
            if num_bytes < header_bytes:
                raise ValueError(
                    f"file {path} is smaller than header_bytes={header_bytes}"
                )

            data_bytes = num_bytes - header_bytes
            if data_bytes % itemsize != 0:
                raise ValueError(
                    f"data size for {path} is not divisible by dtype size {itemsize}"
                )

            num_tokens = data_bytes // itemsize
            shards.append(
                TokenShard(
                    path=path,
                    num_tokens=num_tokens,
                    data_offset_bytes=header_bytes,
                )
            )

        return cls(shards, dtype=dtype)

    def __len__(self):
        return self.total_tokens

    def _find_shard(self, token_index: int) -> tuple[int, int]:
        if token_index < 0 or token_index >= self.total_tokens:
            raise IndexError(
                f"token_index must be in [0, {self.total_tokens}), got {token_index}"
            )

        shard_idx = bisect.bisect_right(self.cumulative_lengths, token_index)
        shard_start = 0
        if shard_idx > 0:
            shard_start = self.cumulative_lengths[shard_idx - 1]

        local_index = token_index - shard_start
        return shard_idx, local_index

    def get_tokens(self, start: int, end: int) -> np.ndarray:
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if end < start:
            raise ValueError(f"end must be >= start, got start={start}, end={end}")
        if end > self.total_tokens:
            raise ValueError(f"end must be <= {self.total_tokens}, got {end}")
        if start == end:
            return np.empty((0,), dtype=self.dtype)

        pieces = []
        cursor = start

        while cursor < end:
            shard_idx, local_start = self._find_shard(cursor)
            shard_global_end = self.cumulative_lengths[shard_idx]
            take_until = min(end, shard_global_end)
            local_end = local_start + (take_until - cursor)

            piece = self.arrays[shard_idx][local_start:local_end]
            pieces.append(np.asarray(piece))
            cursor = take_until

        if len(pieces) == 1:
            return pieces[0]

        return np.concatenate(pieces)


if __name__ == "__main__":
    train_path = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    val_path = Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    paths = [path for path in (train_path, val_path) if path.exists()]

    if not paths:
        raise FileNotFoundError(
            "No Parameter Golf token shards found. Run "
            "`python scripts/download_parameter_golf.py --train-shards 1` first."
        )

    dataset = TokenShardDataset.from_files(paths)
    tokens = dataset.get_tokens(0, 32)

    print("shards:", [str(path) for path in paths])
    print("total_tokens:", len(dataset))
    print("first_32_tokens:", tokens.tolist())
    print("max_first_32:", int(tokens.max()) if len(tokens) else None)

    try:
        from meshtrain.data.tokenizer import load_tokenizer

        tokenizer = load_tokenizer()
        print("decoded_first_32:", tokenizer.decode(tokens.tolist()))
    except Exception as exc:
        print("decode_skipped:", exc)
        
