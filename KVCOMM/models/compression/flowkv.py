from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import math

import torch


@dataclass
class _SegmentBudget:
    index: int
    kind: str
    length: int
    minimum: int
    weight: float
    is_plan_segment: bool = True


class FlowKVSegmentCompressor:
    """Wrap an existing KV compressor with FlowKV-style content isolation."""

    def __init__(
        self,
        *,
        base_cls,
        base_kwargs: Dict[str, Any],
        model_config: Any,
        flowkv_segment_granularity: str = "per_agent",
        flowkv_budget_bias: str = "history_first",
        flowkv_core_reserve: int = 128,
        flowkv_min_agent_budget: int = 32,
    ):
        self.base_cls = base_cls
        self.base_kwargs = dict(base_kwargs)
        self.model_config = model_config
        self.budget = int(base_kwargs.get("budget", 0))
        self.window_size = max(1, int(base_kwargs.get("window_size", 1)))
        self.first_tokens = max(1, int(base_kwargs.get("first_tokens", 1)))
        self.segment_granularity = (flowkv_segment_granularity or "per_agent").lower().strip()
        self.budget_bias = (flowkv_budget_bias or "history_first").lower().strip()
        self.core_reserve = max(0, int(flowkv_core_reserve))
        self.min_agent_budget = max(0, int(flowkv_min_agent_budget))
        self._fallback = self.base_cls(**self.base_kwargs)

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        state_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        plan = self._get_plan(state_key)
        if plan is None or self.segment_granularity != "per_agent":
            return self._fallback.update_kv(
                key_states,
                query_states,
                value_states,
                state_key=state_key,
            )

        seq_len = int(key_states.shape[-2])
        if seq_len <= self.budget or not plan.segments:
            return key_states, value_states

        prompt_segments = [segment for segment in plan.segments if segment.current_token_length > 0]
        prompt_len = sum(int(segment.current_token_length) for segment in prompt_segments)
        prompt_len = min(prompt_len, seq_len)
        generated_len = max(0, seq_len - prompt_len)

        descriptors = self._build_budget_descriptors(prompt_segments, generated_len)
        budgets = self._allocate_budgets(descriptors, self.budget)
        if sum(budgets) <= 0:
            return self._fallback.update_kv(
                key_states,
                query_states,
                value_states,
                state_key=state_key,
            )

        compressed_keys: List[torch.Tensor] = []
        compressed_values: List[torch.Tensor] = []
        cursor = 0
        for descriptor, budget in zip(descriptors, budgets):
            length = int(descriptor.length)
            if length <= 0:
                continue
            block_key = key_states[:, :, cursor : cursor + length, :]
            block_value = value_states[:, :, cursor : cursor + length, :]
            compressed_key, compressed_value = self._compress_block(
                block_key,
                query_states,
                block_value,
                int(budget),
            )
            cursor += length
            compressed_keys.append(compressed_key)
            compressed_values.append(compressed_value)
            if descriptor.is_plan_segment:
                plan.segments[descriptor.index].current_token_length = int(compressed_key.shape[-2])

        if not compressed_keys:
            return self._fallback.update_kv(
                key_states,
                query_states,
                value_states,
                state_key=state_key,
            )
        return torch.cat(compressed_keys, dim=2), torch.cat(compressed_values, dim=2)

    def _get_plan(self, state_key: Optional[str]):
        if not state_key:
            return None
        store = getattr(self.model_config, "_flowkv_prompt_plans", None)
        if not isinstance(store, dict):
            return None
        return store.get(state_key)

    def _build_budget_descriptors(self, prompt_segments, generated_len: int) -> List[_SegmentBudget]:
        descriptors: List[_SegmentBudget] = []
        core_lengths = [
            int(segment.current_token_length)
            for segment in prompt_segments
            if segment.kind == "core"
        ]
        total_core_len = sum(core_lengths)
        remaining_core_reserve = min(self.core_reserve, total_core_len)

        for index, segment in enumerate(prompt_segments):
            length = int(segment.current_token_length)
            minimum = 0
            if segment.kind == "core" and remaining_core_reserve > 0 and total_core_len > 0:
                exact = self.core_reserve * (length / total_core_len)
                minimum = min(length, max(0, math.floor(exact)))
            elif segment.kind == "past":
                minimum = min(length, self.min_agent_budget)
            descriptors.append(
                _SegmentBudget(
                    index=index,
                    kind=segment.kind,
                    length=length,
                    minimum=minimum,
                    weight=self._weight_for_kind(segment.kind),
                    is_plan_segment=True,
                )
            )

        assigned_core = sum(item.minimum for item in descriptors if item.kind == "core")
        core_gap = max(0, remaining_core_reserve - assigned_core)
        if core_gap:
            for item in descriptors:
                if core_gap <= 0:
                    break
                if item.kind != "core":
                    continue
                extra = min(item.length - item.minimum, core_gap)
                item.minimum += extra
                core_gap -= extra

        if generated_len > 0:
            descriptors.append(
                _SegmentBudget(
                    index=-1,
                    kind="generated",
                    length=generated_len,
                    minimum=min(generated_len, self.window_size),
                    weight=self._weight_for_kind("generated"),
                    is_plan_segment=False,
                )
            )
        return descriptors

    def _weight_for_kind(self, kind: str) -> float:
        if self.budget_bias == "current_first":
            weights = {"core": 2.0, "past": 1.0, "current": 2.0, "generated": 2.0}
        elif self.budget_bias == "length_ratio":
            weights = {"core": 1.5, "past": 1.0, "current": 1.0, "generated": 1.0}
        else:
            weights = {"core": 2.5, "past": 2.0, "current": 1.0, "generated": 1.5}
        return float(weights.get(kind, 1.0))

    def _allocate_budgets(self, descriptors: Sequence[_SegmentBudget], total_budget: int) -> List[int]:
        budgets = [0 for _ in descriptors]
        remaining = max(0, int(total_budget))

        # Allocate minimum budgets in priority order: core -> generated -> past -> current.
        priorities = {"core": 0, "generated": 1, "past": 2, "current": 3}
        for idx in sorted(range(len(descriptors)), key=lambda item: priorities.get(descriptors[item].kind, 4)):
            if remaining <= 0:
                break
            descriptor = descriptors[idx]
            allocation = min(descriptor.length, descriptor.minimum, remaining)
            budgets[idx] += allocation
            remaining -= allocation

        if remaining <= 0:
            return budgets

        scores: List[float] = []
        total_score = 0.0
        for budget, descriptor in zip(budgets, descriptors):
            capacity = max(0, descriptor.length - budget)
            score = capacity * descriptor.weight
            scores.append(score)
            total_score += score

        if total_score <= 0:
            return budgets

        exact_shares = []
        for budget, descriptor, score in zip(budgets, descriptors, scores):
            capacity = max(0, descriptor.length - budget)
            if capacity <= 0 or score <= 0:
                exact_shares.append(0.0)
                continue
            exact_shares.append((score / total_score) * remaining)

        allocated = 0
        remainders: List[tuple[float, int]] = []
        for idx, share in enumerate(exact_shares):
            integer_share = min(
                descriptors[idx].length - budgets[idx],
                int(math.floor(share)),
            )
            budgets[idx] += integer_share
            allocated += integer_share
            remainders.append((share - integer_share, idx))

        left = remaining - allocated
        for _, idx in sorted(remainders, reverse=True):
            if left <= 0:
                break
            if budgets[idx] >= descriptors[idx].length:
                continue
            budgets[idx] += 1
            left -= 1
        return budgets

    def _compress_block(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        length = int(key_states.shape[-2])
        if budget <= 0:
            empty_key = key_states[:, :, :0, :]
            empty_value = value_states[:, :, :0, :]
            return empty_key, empty_value
        if length <= budget:
            return key_states, value_states

        local_kwargs = dict(self.base_kwargs)
        local_kwargs["budget"] = int(budget)
        try:
            compressor = self.base_cls(**local_kwargs)
            return compressor.update_kv(
                key_states,
                query_states,
                value_states,
                state_key=None,
            )
        except Exception:
            if budget >= length:
                return key_states, value_states
            return (
                key_states[:, :, -budget:, :],
                value_states[:, :, -budget:, :],
            )
