from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence


@dataclass
class PromptSegment:
    kind: str
    text: str
    agent_id: Optional[str] = None
    role: Optional[str] = None
    message_role: str = "user"
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    token_length: int = 0
    current_token_length: int = 0

    @classmethod
    def from_any(cls, value: "PromptSegment | dict[str, Any]") -> "PromptSegment":
        if isinstance(value, cls):
            return cls(**value.__dict__)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported prompt segment payload: {type(value)!r}")


@dataclass
class FlowKVContentPlan:
    prompt_text: str
    prompt_token_length: int
    segments: List[PromptSegment]

    def current_prompt_token_length(self) -> int:
        return sum(max(0, int(segment.current_token_length)) for segment in self.segments)


def _locate_token_span(
    offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    hit_indices = [
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_start < char_end and token_end > char_start
    ]
    if not hit_indices:
        return 0, 0
    return hit_indices[0], hit_indices[-1] + 1


def _fallback_token_span(tokenizer: Any, prompt_text: str, char_start: int, char_end: int) -> tuple[int, int]:
    prefix = tokenizer.encode(prompt_text[:char_start], add_special_tokens=False)
    inclusive = tokenizer.encode(prompt_text[:char_end], add_special_tokens=False)
    return len(prefix), len(inclusive)


def _iter_segments(segments: Optional[Iterable[PromptSegment | dict[str, Any]]]) -> List[PromptSegment]:
    if segments is None:
        return []
    return [PromptSegment.from_any(segment) for segment in segments]


def _find_segment_char_span(prompt_text: str, text: str, cursor: int) -> tuple[int, int]:
    char_start = prompt_text.find(text, cursor)
    if char_start < 0:
        char_start = prompt_text.find(text)
    if char_start >= 0:
        return char_start, char_start + len(text)

    stripped_text = text.strip()
    if not stripped_text:
        return cursor, cursor

    if stripped_text != text:
        char_start = prompt_text.find(stripped_text, cursor)
        if char_start < 0:
            char_start = prompt_text.find(stripped_text)
        if char_start >= 0:
            return char_start, char_start + len(stripped_text)

    raise ValueError("Unable to locate prompt segment text in prompt text.")


def build_flowkv_content_plan(
    *,
    tokenizer: Any,
    prompt_text: str,
    segments: Optional[Iterable[PromptSegment | dict[str, Any]]],
    prompt_token_length: Optional[int] = None,
) -> Optional[FlowKVContentPlan]:
    normalised_segments = _iter_segments(segments)
    if not normalised_segments:
        return None

    encoded_with_offsets = None
    offsets: Sequence[tuple[int, int]] = []
    try:
        encoded_with_offsets = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded_with_offsets.get("offset_mapping", [])
    except Exception:
        offsets = []

    if prompt_token_length is None:
        if encoded_with_offsets is not None and "input_ids" in encoded_with_offsets:
            prompt_token_length = len(encoded_with_offsets["input_ids"])
        else:
            prompt_token_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    cursor = 0
    for segment in normalised_segments:
        text = segment.text or ""
        if not text:
            segment.char_start = cursor
            segment.char_end = cursor
            segment.token_start = 0
            segment.token_end = 0
            segment.token_length = 0
            segment.current_token_length = 0
            continue

        try:
            char_start, char_end = _find_segment_char_span(prompt_text, text, cursor)
        except ValueError as exc:
            raise ValueError(
                "Unable to map prompt segment into prompt text. "
                f"kind={segment.kind!r} agent_id={segment.agent_id!r} role={segment.role!r}"
            ) from exc
        cursor = char_end
        segment.char_start = char_start
        segment.char_end = char_end

        if char_start == char_end:
            segment.token_start = 0
            segment.token_end = 0
            segment.token_length = 0
            segment.current_token_length = 0
            continue

        if offsets:
            token_start, token_end = _locate_token_span(offsets, char_start, char_end)
        else:
            token_start, token_end = _fallback_token_span(tokenizer, prompt_text, char_start, char_end)
        segment.token_start = token_start
        segment.token_end = token_end
        segment.token_length = max(0, token_end - token_start)
        segment.current_token_length = segment.token_length

    return FlowKVContentPlan(
        prompt_text=prompt_text,
        prompt_token_length=int(prompt_token_length),
        segments=normalised_segments,
    )
