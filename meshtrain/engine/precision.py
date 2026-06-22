from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

from meshtrain.engine.config import PrecisionConfig


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16

    raise ValueError(f"unsupported precision dtype: {dtype}")


def autocast_context(
    config: PrecisionConfig,
    *,
    device: torch.device,
) -> ContextManager:
    if not config.autocast:
        return nullcontext()

    if config.dtype == "fp32":
        return nullcontext()

    if device.type != "cuda":
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=resolve_torch_dtype(config.dtype),
    )


def create_grad_scaler(
    config: PrecisionConfig,
    *,
    device: torch.device,
) -> torch.amp.GradScaler | None:
    if not config.grad_scaler:
        return None

    if config.dtype != "fp16":
        raise ValueError("GradScaler should only be enabled for fp16")

    if device.type != "cuda":
        raise ValueError("GradScaler requires a CUDA device")

    return torch.amp.GradScaler(device.type)


def maybe_cast_model(
    model: torch.nn.Module,
    config: PrecisionConfig,
) -> torch.nn.Module:
    if config.dtype == "fp32":
        return model

    return model.to(dtype=resolve_torch_dtype(config.dtype))
