from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSamplePlan:
    global_batch_id: int
    global_sample_ids: list[int]
    local_sample_ids: list[int]
    local_batch_range: tuple[int, int]


class DPSampler:
    def __init__(
        self,
        num_examples: int,
        global_batch_size: int,
        dp_rank: int = 0,
        dp_size: int = 1,
        drop_last: bool = True,
    ):
        if num_examples < 1:
            raise ValueError(f"num_examples must be at least 1, got {num_examples}")
        if global_batch_size < 1:
            raise ValueError(
                f"global_batch_size must be at least 1, got {global_batch_size}"
            )
        if dp_size < 1:
            raise ValueError(f"dp_size must be at least 1, got {dp_size}")
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"dp_rank must be in [0, {dp_size}), got {dp_rank}")
        if global_batch_size % dp_size != 0:
            raise ValueError(
                f"global_batch_size {global_batch_size} must divide evenly by dp_size {dp_size}"
            )
        if not drop_last:
            raise NotImplementedError("drop_last=False is not supported yet")

        self.num_examples = num_examples
        self.global_batch_size = global_batch_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.drop_last = drop_last
        self.local_batch_size = global_batch_size // dp_size
        self.num_global_batches = num_examples // global_batch_size

        if self.num_global_batches < 1:
            raise ValueError(
                f"num_examples={num_examples} is smaller than global_batch_size={global_batch_size}"
            )

    def __len__(self) -> int:
        return self.num_global_batches

    def plan_batch(self, global_batch_id: int) -> BatchSamplePlan:
        if global_batch_id < 0 or global_batch_id >= self.num_global_batches:
            raise IndexError(
                f"global_batch_id must be in [0, {self.num_global_batches}), got {global_batch_id}"
            )

        global_start = global_batch_id * self.global_batch_size
        global_end = global_start + self.global_batch_size
        global_sample_ids = list(range(global_start, global_end))

        local_start = self.dp_rank * self.local_batch_size
        local_end = local_start + self.local_batch_size
        local_sample_ids = global_sample_ids[local_start:local_end]

        return BatchSamplePlan(
            global_batch_id=global_batch_id,
            global_sample_ids=global_sample_ids,
            local_sample_ids=local_sample_ids,
            local_batch_range=(local_start, local_end),
        )


if __name__ == "__main__":
    num_examples = 32
    global_batch_size = 8
    dp_size = 2

    for dp_rank in range(dp_size):
        sampler = DPSampler(
            num_examples=num_examples,
            global_batch_size=global_batch_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        plan = sampler.plan_batch(0)
        print(f"dp_rank={dp_rank}")
        print("  global_sample_ids:", plan.global_sample_ids)
        print("  local_batch_range:", plan.local_batch_range)
        print("  local_sample_ids:", plan.local_sample_ids)
