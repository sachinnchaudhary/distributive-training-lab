from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from meshtrain.engine.config import CheckpointConfig, EngineConfig


@dataclass(frozen=True)
class CheckpointState:
    step: int
    path: Path
    saved: bool


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be >= 0")

    return Path(output_dir) / f"step_{step:08d}"


def _config_to_dict(config: EngineConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None

    if is_dataclass(config):
        return asdict(config)

    raise TypeError("config must be an EngineConfig dataclass or None")


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: EngineConfig | None,
    step: int,
    checkpoint_config: CheckpointConfig,
    rank: int,
) -> CheckpointState:
    path = checkpoint_dir(checkpoint_config.output_dir, step)

    if rank != 0:
        return CheckpointState(step=step, path=path, saved=False)

    path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "config": _config_to_dict(config),
    }

    if checkpoint_config.save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path / "checkpoint.pt")

    return CheckpointState(step=step, path=path, saved=True)


def load_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    map_location: str | torch.device | None = None,
    load_optimizer: bool = True,
) -> int:
    path = Path(path)
    checkpoint_path = path / "checkpoint.pt" if path.is_dir() else path

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(payload["model"])

    if load_optimizer and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    return int(payload["step"])


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    checkpoint_dirs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    ]

    if not checkpoint_dirs:
        return None

    return max(checkpoint_dirs, key=lambda path: path.name)
