from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from meshtrain.core.distributed.groups import ParallelGroups
from meshtrain.core.distributed.placement import split_range


@dataclass(frozen=True)
class ExpertShard:
    start: int
    end: int
    num_experts: int
    ep_rank: int
    ep_size: int

    @property
    def num_local_experts(self) -> int:
        return self.end - self.start

    def owns(self, expert_id: int) -> bool:
        return self.start <= expert_id < self.end

    def local_expert_index(self, expert_id: int) -> int:
        if not self.owns(expert_id):
            raise ValueError(
                f"expert {expert_id} is not owned by shard [{self.start}, {self.end})"
            )
        return expert_id - self.start


@dataclass(frozen=True)
class ExpertRoutingPlan:
    sorted_token_indices: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_dest_ep_ranks: torch.Tensor
    send_counts: torch.Tensor


@dataclass(frozen=True)
class ExpertDispatch:
    received_tokens: torch.Tensor
    received_expert_ids: torch.Tensor
    received_source_ep_ranks: torch.Tensor
    received_source_token_indices: torch.Tensor
    send_counts: torch.Tensor
    recv_counts: torch.Tensor


def _ep_size(groups: ParallelGroups) -> int:
    return len(groups.ep_ranks)


def _ep_rank(groups: ParallelGroups) -> int:
    return groups.ep_ranks.index(groups.rank)


def _ep_is_active(groups: ParallelGroups) -> bool:
    return groups.ep_group is not None and len(groups.ep_ranks) > 1


def expert_parallel_range(
    num_experts: int,
    groups: ParallelGroups,
) -> ExpertShard:
    if num_experts < 1:
        raise ValueError("num_experts must be >= 1")

    ep_rank = _ep_rank(groups)
    ep_size = _ep_size(groups)

    index_range = split_range(
        size=num_experts,
        parts=ep_size,
        index=ep_rank,
        require_even=False,
    )

    return ExpertShard(
        start=index_range.start,
        end=index_range.end,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )


def expert_owner_ep_rank(
    expert_id: int,
    num_experts: int,
    ep_size: int,
) -> int:
    if num_experts < 1:
        raise ValueError("num_experts must be >= 1")
    if ep_size < 1:
        raise ValueError("ep_size must be >= 1")
    if expert_id < 0 or expert_id >= num_experts:
        raise ValueError(f"expert_id must be in [0, {num_experts}), got {expert_id}")

    for ep_rank in range(ep_size):
        index_range = split_range(
            size=num_experts,
            parts=ep_size,
            index=ep_rank,
            require_even=False,
        )
        if index_range.start <= expert_id < index_range.end:
            return ep_rank

    raise RuntimeError("unreachable expert owner lookup")


def expert_owner_rank(
    expert_id: int,
    num_experts: int,
    groups: ParallelGroups,
) -> int:
    owner_ep_rank = expert_owner_ep_rank(
        expert_id,
        num_experts,
        _ep_size(groups),
    )
    return groups.ep_ranks[owner_ep_rank]


def build_expert_routing_plan(
    expert_ids: torch.Tensor,
    *,
    num_experts: int,
    groups: ParallelGroups,
) -> ExpertRoutingPlan:
    if expert_ids.ndim != 1:
        raise ValueError("expert_ids must be a 1D tensor")

    ep_size = _ep_size(groups)
    dest_ep_ranks = [
        expert_owner_ep_rank(int(expert_id), num_experts, ep_size)
        for expert_id in expert_ids.tolist()
    ]
    dest_ep_ranks_tensor = torch.tensor(
        dest_ep_ranks,
        device=expert_ids.device,
        dtype=torch.long,
    )

    sorted_dest_ep_ranks, sorted_token_indices = torch.sort(
        dest_ep_ranks_tensor,
        stable=True,
    )
    sorted_expert_ids = expert_ids.index_select(0, sorted_token_indices)

    send_counts = torch.bincount(
        sorted_dest_ep_ranks,
        minlength=ep_size,
    ).to(torch.long)

    return ExpertRoutingPlan(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_ids=sorted_expert_ids,
        sorted_dest_ep_ranks=sorted_dest_ep_ranks,
        send_counts=send_counts,
    )


def _exchange_counts(
    send_counts: torch.Tensor,
    groups: ParallelGroups,
) -> torch.Tensor:
    recv_counts = torch.empty_like(send_counts)

    if _ep_is_active(groups):
        dist.all_to_all_single(
            recv_counts,
            send_counts,
            group=groups.ep_group,
        )
    else:
        recv_counts.copy_(send_counts)

    return recv_counts


def _all_to_all_by_counts(
    send_tensor: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    groups: ParallelGroups,
) -> torch.Tensor:
    output_shape = (int(recv_counts.sum().item()), *send_tensor.shape[1:])
    recv_tensor = torch.empty(
        output_shape,
        device=send_tensor.device,
        dtype=send_tensor.dtype,
    )

    if _ep_is_active(groups):
        dist.all_to_all_single(
            recv_tensor,
            send_tensor.contiguous(),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=groups.ep_group,
        )
    else:
        recv_tensor.copy_(send_tensor)

    return recv_tensor


def dispatch_tokens_to_experts(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    num_experts: int,
    groups: ParallelGroups,
) -> ExpertDispatch:
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [num_tokens, hidden_dim]")
    if expert_ids.ndim != 1:
        raise ValueError("expert_ids must have shape [num_tokens]")
    if tokens.shape[0] != expert_ids.shape[0]:
        raise ValueError("tokens and expert_ids must have the same first dimension")

    plan = build_expert_routing_plan(
        expert_ids,
        num_experts=num_experts,
        groups=groups,
    )
    send_counts = plan.send_counts
    recv_counts = _exchange_counts(send_counts, groups)

    send_tokens = tokens.index_select(0, plan.sorted_token_indices).contiguous()
    send_expert_ids = plan.sorted_expert_ids.to(torch.long).contiguous()
    send_source_token_indices = plan.sorted_token_indices.to(torch.long).contiguous()
    send_source_ep_ranks = torch.full_like(
        send_source_token_indices,
        fill_value=_ep_rank(groups),
    )

    received_tokens = _all_to_all_by_counts(
        send_tokens,
        send_counts,
        recv_counts,
        groups,
    )
    received_expert_ids = _all_to_all_by_counts(
        send_expert_ids,
        send_counts,
        recv_counts,
        groups,
    )
    received_source_token_indices = _all_to_all_by_counts(
        send_source_token_indices,
        send_counts,
        recv_counts,
        groups,
    )
    received_source_ep_ranks = _all_to_all_by_counts(
        send_source_ep_ranks,
        send_counts,
        recv_counts,
        groups,
    )

    return ExpertDispatch(
        received_tokens=received_tokens,
        received_expert_ids=received_expert_ids,
        received_source_ep_ranks=received_source_ep_ranks,
        received_source_token_indices=received_source_token_indices,
        send_counts=send_counts,
        recv_counts=recv_counts,
    )


def combine_expert_outputs(
    expert_outputs: torch.Tensor,
    dispatch: ExpertDispatch,
    *,
    original_num_tokens: int,
    groups: ParallelGroups,
) -> torch.Tensor:
    if expert_outputs.shape[0] != dispatch.received_tokens.shape[0]:
        raise ValueError("expert_outputs must match dispatched token count")

    ep_size = _ep_size(groups)
    source_ep_ranks = dispatch.received_source_ep_ranks.to(torch.long)
    sorted_source_ep_ranks, sorted_indices = torch.sort(source_ep_ranks, stable=True)

    send_counts = torch.bincount(
        sorted_source_ep_ranks,
        minlength=ep_size,
    ).to(torch.long)
    recv_counts = _exchange_counts(send_counts, groups)

    send_outputs = expert_outputs.index_select(0, sorted_indices).contiguous()
    send_token_indices = dispatch.received_source_token_indices.index_select(
        0,
        sorted_indices,
    ).to(torch.long).contiguous()

    returned_outputs = _all_to_all_by_counts(
        send_outputs,
        send_counts,
        recv_counts,
        groups,
    )
    returned_token_indices = _all_to_all_by_counts(
        send_token_indices,
        send_counts,
        recv_counts,
        groups,
    )

    combined = torch.empty(
        (original_num_tokens, *expert_outputs.shape[1:]),
        device=expert_outputs.device,
        dtype=expert_outputs.dtype,
    )
    combined.index_copy_(0, returned_token_indices, returned_outputs)
    return combined


def run_local_experts(
    received_tokens: torch.Tensor,
    received_expert_ids: torch.Tensor,
    experts: nn.ModuleList,
    shard: ExpertShard,
) -> torch.Tensor:
    if len(experts) != shard.num_local_experts:
        raise ValueError("number of local experts must match expert shard")

    outputs = torch.empty_like(received_tokens)
    for global_expert_id in range(shard.start, shard.end):
        mask = received_expert_ids == global_expert_id
        if not bool(mask.any()):
            continue

        local_index = shard.local_expert_index(global_expert_id)
        outputs[mask] = experts[local_index](received_tokens[mask])

    return outputs
