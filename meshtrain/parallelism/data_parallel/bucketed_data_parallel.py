from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.parallelism.data_parallel.data_parallelism import (
    ReplicaCheck,
    check_replicas_match,
)


@dataclass(frozen=True)
class BucketedDPConfig:
    bucket_size_mb: int = 25
    async_all_reduce: bool = False
    overlap_backward: bool = False


@dataclass(frozen=True)
class GradientBucket:
    bucket_id: int
    parameters: list[nn.Parameter]
    dtype: torch.dtype
    device: torch.device
    numel: int

@dataclass(frozen=True)
class PendingBucketAllReduce:
    bucket: GradientBucket
    flat_grad: torch.Tensor
    work: dist.Work


def _dp_is_active(groups: ParallelGroups) -> bool:
    return groups.dp_group is not None and len(groups.dp_ranks) > 1


def _bucket_size_bytes(config: BucketedDPConfig) -> int:
    if config.bucket_size_mb <= 0:
        raise ValueError(f"bucket_size_mb must be positive, got {config.bucket_size_mb}")
    return config.bucket_size_mb * 1024 * 1024


def _parameter_nbytes(parameter: nn.Parameter) -> int:
    return parameter.numel() * parameter.element_size()


def _make_bucket(bucket_id: int, parameters: list[nn.Parameter]) -> GradientBucket:
    if not parameters:
        raise ValueError("cannot create an empty gradient bucket")

    first = parameters[0]
    return GradientBucket(
        bucket_id=bucket_id,
        parameters=list(parameters),
        dtype=first.dtype,
        device=first.device,
        numel=sum(parameter.numel() for parameter in parameters),
    )


def build_gradient_buckets(
    model: nn.Module,
    config: BucketedDPConfig,
) -> list[GradientBucket]:
    max_bucket_bytes = _bucket_size_bytes(config)
    buckets: list[GradientBucket] = []

    current_params: list[nn.Parameter] = []
    current_bytes = 0
    current_dtype: torch.dtype | None = None
    current_device: torch.device | None = None

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        param_dtype = parameter.dtype
        param_device = parameter.device
        param_bytes = _parameter_nbytes(parameter)

        should_start_new_bucket = False
        if current_params:
            should_start_new_bucket = (
                param_dtype != current_dtype
                or param_device != current_device
                or current_bytes + param_bytes > max_bucket_bytes
            )

        if should_start_new_bucket:
            buckets.append(_make_bucket(len(buckets), current_params))
            current_params = []
            current_bytes = 0
            current_dtype = None
            current_device = None

        if not current_params:
            current_dtype = param_dtype
            current_device = param_device

        current_params.append(parameter)
        current_bytes += param_bytes

    if current_params:
        buckets.append(_make_bucket(len(buckets), current_params))

    return buckets


class BucketedDataParallel:
    def __init__(
        self,  
        model: nn.Module,
        groups: ParallelGroups,
        config: BucketedDPConfig | None = None,
    ):
        self.model = model
        self.groups = groups
        self.config = config or BucketedDPConfig()
        self._pending: list[PendingBucketAllReduce] = [] 

        if self.config.overlap_backward and not self.config.async_all_reduce:
            raise ValueError("overlap_backward requires async_all_reduce=True")

        self.buckets = build_gradient_buckets(model, self.config)
        self._parameter_to_bucket_id: dict[int, int] = {}
        self._bucket_ready_counts = [0 for _ in self.buckets]
        self._bucket_launched = [False for _ in self.buckets]
        self._hooks = []

        for bucket in self.buckets:
            for parameter in bucket.parameters:
                self._parameter_to_bucket_id[id(parameter)] = bucket.bucket_id

        if self.config.overlap_backward:
            self._register_backward_hooks()

    def broadcast_parameters(self, src_rank: int | None = None) -> None:
        if not _dp_is_active(self.groups):
            return

        if src_rank is None:
            src_rank = self.groups.dp_ranks[0]

        for parameter in self.model.parameters():
            dist.broadcast(
                parameter.data,
                src=src_rank,
                group=self.groups.dp_group,
            )

    def sync_gradients(self) -> None:
        if not _dp_is_active(self.groups):
            return
        if self.config.overlap_backward:
            return
        if self._pending:
            raise RuntimeError(
                "cannot start a new gradient sync before finalize_backward()"
            )

        for bucket in self.buckets:
            self._launch_bucket_all_reduce(bucket)

    def prepare_for_backward(self) -> None:
        if not _dp_is_active(self.groups):
            return
        if self._pending:
            raise RuntimeError(
                "cannot start backward before previous communication is finalized"
            )

        self._bucket_ready_counts = [0 for _ in self.buckets]
        self._bucket_launched = [False for _ in self.buckets]

    def _flatten_bucket_grads(self, bucket: GradientBucket) -> torch.Tensor:
        for parameter in bucket.parameters:
            if parameter.grad is None:
                raise RuntimeError(
                    "all parameters in a gradient bucket must have gradients before sync"
                )

        flat_grad = torch.empty(
            bucket.numel,
            dtype=bucket.dtype,
            device=bucket.device,
        )

        offset = 0
        for parameter in bucket.parameters:
            grad = parameter.grad
            assert grad is not None
            numel = grad.numel()
            flat_grad[offset : offset + numel].copy_(grad.reshape(-1))
            offset += numel

        return flat_grad

    def _copy_bucket_grads_from_flat(
        self,
        bucket: GradientBucket,
        flat_grad: torch.Tensor,
    ) -> None:
        offset = 0
        for parameter in bucket.parameters:
            grad = parameter.grad
            assert grad is not None
            numel = grad.numel()
            grad.copy_(flat_grad[offset : offset + numel].view_as(grad))
            offset += numel

    def _launch_bucket_all_reduce(self, bucket: GradientBucket) -> None:
        flat_grad = self._flatten_bucket_grads(bucket)

        if self.config.async_all_reduce:
            work = dist.all_reduce(
                flat_grad,
                op=dist.ReduceOp.SUM,
                group=self.groups.dp_group,
                async_op=True,
            )
            self._pending.append(
                PendingBucketAllReduce(
                    bucket=bucket,
                    flat_grad=flat_grad,
                    work=work,
                )
            )
            return

        dist.all_reduce(
            flat_grad,
            op=dist.ReduceOp.SUM,
            group=self.groups.dp_group,
        )
        self._copy_bucket_grads_from_flat(bucket, flat_grad)

    def _register_backward_hooks(self) -> None:
        for bucket in self.buckets:
            for parameter in bucket.parameters:
                if not hasattr(parameter, "register_post_accumulate_grad_hook"):
                    raise RuntimeError(
                        "overlap_backward requires register_post_accumulate_grad_hook"
                    )

                hook = parameter.register_post_accumulate_grad_hook(
                    self._make_post_accumulate_hook(parameter)
                )
                self._hooks.append(hook)

    def _make_post_accumulate_hook(self, parameter: nn.Parameter):
        bucket_id = self._parameter_to_bucket_id[id(parameter)]

        def hook(_parameter: torch.Tensor) -> None:
            self._mark_parameter_ready(bucket_id)

        return hook

    def _mark_parameter_ready(self, bucket_id: int) -> None:
        if not self.config.overlap_backward or not _dp_is_active(self.groups):
            return

        self._bucket_ready_counts[bucket_id] += 1
        bucket = self.buckets[bucket_id]

        if self._bucket_ready_counts[bucket_id] == len(bucket.parameters):
            self._launch_ready_bucket(bucket)
        elif self._bucket_ready_counts[bucket_id] > len(bucket.parameters):
            raise RuntimeError(f"bucket {bucket_id} received too many ready signals")

    def _launch_ready_bucket(self, bucket: GradientBucket) -> None:
        bucket_id = bucket.bucket_id
        if self._bucket_launched[bucket_id]:
            raise RuntimeError(f"bucket {bucket_id} was launched more than once")

        self._bucket_launched[bucket_id] = True
        self._launch_bucket_all_reduce(bucket)

    def finalize_backward(self) -> None:
        if not _dp_is_active(self.groups):
            return

        if self.config.overlap_backward:
            for bucket in self.buckets:
                if not self._bucket_launched[bucket.bucket_id]:
                    raise RuntimeError(
                        f"bucket {bucket.bucket_id} was not launched; a gradient may be missing"
                    )

        if not self._pending:
            return

        for pending in self._pending:
            pending.work.wait()
            self._copy_bucket_grads_from_flat(pending.bucket, pending.flat_grad)

        self._pending.clear()

    def check_replicas_match(self, *, atol: float = 1e-5) -> ReplicaCheck:
        return check_replicas_match(self.model, self.groups, atol=atol)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
