from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "willdepueoai/parameter-golf"
DEFAULT_VARIANT = "sp1024"


def dataset_dir_for_variant(variant: str) -> Path:
    if variant == "sp1024":
        return Path("data/datasets/fineweb10B_sp1024")
    raise ValueError(f"unsupported variant: {variant}")


def download_file(repo_id: str, remote_path: str, local_path: Path) -> None:
  
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"exists {local_path}")
        return

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_path,
    )
    local_path.write_bytes(Path(downloaded).read_bytes())


def build_file_list(variant: str, train_shards: int) -> list[tuple[str, Path]]:
    if train_shards < 0:
        raise ValueError(f"train_shards must be non-negative, got {train_shards}")

    dataset_dir = dataset_dir_for_variant(variant)
    tokenizer_dir = Path("data/tokenizers")

    files = [
        (
            "datasets/tokenizers/fineweb_1024_bpe.model",
            tokenizer_dir / "fineweb_1024_bpe.model",
        ),
        (
            "datasets/tokenizers/fineweb_1024_bpe.vocab",
            tokenizer_dir / "fineweb_1024_bpe.vocab",
        ),
        (
            "datasets/datasets/fineweb10B_sp1024/fineweb_val_000000.bin",
            dataset_dir / "fineweb_val_000000.bin",
        ),
    ]

    for i in range(train_shards):
        name = f"fineweb_train_{i:06d}.bin"
        files.append(
            (
                f"datasets/datasets/fineweb10B_sp1024/{name}",
                dataset_dir / name,
            )
        )

    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--train-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = build_file_list(args.variant, args.train_shards)

    for remote_path, local_path in files:
        print(f"downloading {remote_path} -> {local_path}")
        download_file(args.repo_id, remote_path, local_path)

    print("done")


if __name__ == "__main__":
    main()
