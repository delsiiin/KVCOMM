from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional

import numpy as np
try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for lightweight test environments
    logger = logging.getLogger(__name__)

_SESSION_INDEX_LOCK = Lock()


def build_heatmap_tag(
    attn_heatmap_mode: bool,
    attn_heatmap_layer: Optional[int],
) -> str:
    """Return a stable tag segment for attention visualization artifacts."""
    if not attn_heatmap_mode:
        return "no-heatmap"
    if attn_heatmap_layer is None:
        return "heatmap-all-layers"
    return f"heatmap-l{int(attn_heatmap_layer)}"


def sanitize_filename_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "unknown"


def _log_artifact(payload: Dict[str, Any]) -> None:
    if hasattr(logger, "opt"):
        logger.opt(colors=True).info(
            "<blue>[ATTN VIS]</blue> {}",
            json.dumps(payload, ensure_ascii=False),
        )
        return
    logger.info("[ATTN VIS] %s", json.dumps(payload, ensure_ascii=False))


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        text = ""
    if text:
        return text
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        return str(token_id)


def decode_token_texts(tokenizer: Any, token_ids: Iterable[int]) -> list[str]:
    return [_decode_token(tokenizer, int(token_id)) for token_id in token_ids]


def compute_matrix_stats(matrix: Any) -> Dict[str, float]:
    matrix_np = np.asarray(matrix, dtype=np.float32)
    if matrix_np.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p1": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(np.min(matrix_np)),
        "max": float(np.max(matrix_np)),
        "mean": float(np.mean(matrix_np)),
        "p1": float(np.percentile(matrix_np, 1)),
        "p95": float(np.percentile(matrix_np, 95)),
        "p99": float(np.percentile(matrix_np, 99)),
    }


def compute_token_scores(matrix: Any) -> list[float]:
    matrix_np = np.asarray(matrix, dtype=np.float32)
    if matrix_np.ndim != 2 or matrix_np.shape[1] == 0:
        return []
    return matrix_np.mean(axis=0, dtype=np.float32).astype(np.float32).tolist()


def _normalise_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _load_session_index(index_path: Path, run_tag: str) -> Dict[str, Any]:
    if not index_path.exists():
        return {"run_tag": run_tag, "calls": []}
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        logger.warning("Failed to read session index {}, rebuilding.", index_path)
        return {"run_tag": run_tag, "calls": []}
    if not isinstance(payload, dict):
        return {"run_tag": run_tag, "calls": []}
    payload.setdefault("run_tag", run_tag)
    payload.setdefault("calls", [])
    return payload


def export_attention_visualization(
    *,
    matrices: Dict[int, Any],
    output_dir: str | Path,
    run_tag: str,
    request_uid: Optional[str],
    round_index: Optional[int],
    agent_id: Optional[str],
    agent_name: Optional[str],
    agent_role: Optional[str],
    prompt_text: str,
    response_text: str,
    input_token_ids: list[int],
    input_token_texts: list[str],
    input_token_len: Optional[int] = None,
    output_token_len: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Persist one agent call as structured attention visualization artifacts."""
    if not matrices:
        logger.warning(
            "Attention visualization export skipped because no matrices were captured for request={} agent={}.",
            request_uid,
            agent_id,
        )
        return None

    output_path = Path(output_dir).expanduser()
    safe_run_tag = sanitize_filename_component(run_tag)
    run_dir = output_path / safe_run_tag
    safe_request_uid = sanitize_filename_component(request_uid or "no-request")
    safe_round = "na" if round_index is None else str(int(round_index))
    safe_agent_id = sanitize_filename_component(agent_id or "unknown-agent")
    request_dir = run_dir / "requests" / f"req-{safe_request_uid}" / f"round-{safe_round}"
    request_dir.mkdir(parents=True, exist_ok=True)

    json_path = request_dir / f"agent-{safe_agent_id}.json"
    npz_path = request_dir / f"agent-{safe_agent_id}.npz"
    index_path = run_dir / "session_index.json"

    serialisable_matrices: Dict[str, np.ndarray] = {}
    matrix_stats: Dict[str, Dict[str, float]] = {}
    available_layers: list[int] = sorted(int(layer_idx) for layer_idx in matrices.keys())

    for layer_idx in available_layers:
        matrix_np = np.asarray(matrices[layer_idx], dtype=np.float32)
        if matrix_np.ndim != 2:
            logger.warning(
                "Skipping invalid attention matrix for request={} agent={} layer={} with shape {}.",
                request_uid,
                agent_id,
                layer_idx,
                tuple(matrix_np.shape),
            )
            continue
        serialisable_matrices[f"layer_{layer_idx}"] = matrix_np
        matrix_stats[str(layer_idx)] = compute_matrix_stats(matrix_np)

    if not serialisable_matrices:
        logger.warning(
            "Attention visualization export skipped because no 2D matrices were serialisable for request={} agent={}.",
            request_uid,
            agent_id,
        )
        return None

    np.savez_compressed(npz_path, **serialisable_matrices)

    detail_payload = {
        "run_tag": run_tag,
        "request_uid": request_uid,
        "round_index": round_index,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_role": agent_role,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "token_ids": [int(token_id) for token_id in input_token_ids],
        "tokens": list(input_token_texts),
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "available_layers": available_layers,
        "matrix_shape": list(next(iter(serialisable_matrices.values())).shape),
        "matrix_stats": matrix_stats,
        "npz_path": _normalise_relpath(npz_path, run_dir),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(detail_payload, handle, ensure_ascii=False, indent=2)

    index_entry = {
        "request_uid": request_uid,
        "round_index": round_index,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_role": agent_role,
        "available_layers": available_layers,
        "json_path": _normalise_relpath(json_path, run_dir),
        "npz_path": _normalise_relpath(npz_path, run_dir),
    }

    with _SESSION_INDEX_LOCK:
        run_dir.mkdir(parents=True, exist_ok=True)
        index_payload = _load_session_index(index_path, run_tag)
        calls = index_payload.setdefault("calls", [])
        replaced = False
        for idx, existing in enumerate(calls):
            if (
                existing.get("request_uid") == request_uid
                and existing.get("round_index") == round_index
                and existing.get("agent_id") == agent_id
            ):
                calls[idx] = index_entry
                replaced = True
                break
        if not replaced:
            calls.append(index_entry)
        calls.sort(
            key=lambda item: (
                str(item.get("request_uid") or ""),
                int(item.get("round_index") or -1),
                str(item.get("agent_id") or ""),
            )
        )
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(index_payload, handle, ensure_ascii=False, indent=2)

    artifacts = {
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "npz_path": str(npz_path),
        "available_layers": available_layers,
    }
    _log_artifact(artifacts)
    return artifacts
