"""Legacy KVCOMM compatibility shim.

The project now runs in default execution mode only. This module remains as a
minimal placeholder so older imports do not fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class _RequestState:
    request_uid: str


class KVCOMMEngine:
    """Disabled compatibility surface for removed KV reuse pipeline."""

    anchors: Dict[str, Any] = {}
    anchor_dict: Dict[str, Any] = {}
    anchor_len_dict: Dict[str, Any] = {}
    anchor_info_dict: Dict[str, Any] = {}
    weight_dict: Dict[str, Any] = {}
    global_anchor_info_dict: Dict[str, Any] = {}
    _request_lock = None
    _request_states: Dict[str, _RequestState] = {}
    _active_requests = set()
    _staged_commits = []
    _DISABLED_ERROR = "KV reuse pipeline has been removed; only default mode is supported."

    def __init__(self, llm: Any):
        self.llm = llm

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        return None

    def resolve_request_state(self, request_uid: str) -> _RequestState:
        raise RuntimeError(self._DISABLED_ERROR)

    def get_request_state(self, request_uid: str) -> _RequestState:
        raise RuntimeError(self._DISABLED_ERROR)
