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
ATTENTION_STAGE_PREFILL = "prefill"
ATTENTION_STAGE_GENERATION = "generation"
ATTENTION_STAGE_COMBINED = "combined"
ATTENTION_CAPTURE_STAGES = (
    ATTENTION_STAGE_PREFILL,
    ATTENTION_STAGE_GENERATION,
)
ATTENTION_VIEW_STAGES = (
    ATTENTION_STAGE_PREFILL,
    ATTENTION_STAGE_GENERATION,
    ATTENTION_STAGE_COMBINED,
)


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
    finite_values = matrix_np[np.isfinite(matrix_np)]
    if finite_values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p1": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "p1": float(np.percentile(finite_values, 1)),
        "p95": float(np.percentile(finite_values, 95)),
        "p99": float(np.percentile(finite_values, 99)),
    }


def compute_token_scores(matrix: Any) -> list[float]:
    matrix_np = np.asarray(matrix, dtype=np.float32)
    if matrix_np.ndim != 2 or matrix_np.shape[1] == 0:
        return []
    finite_mask = np.isfinite(matrix_np)
    sums = np.where(finite_mask, matrix_np, 0.0).sum(axis=0, dtype=np.float32)
    counts = finite_mask.sum(axis=0, dtype=np.int32)
    scores = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float32),
        where=counts > 0,
    )
    scores = np.asarray(scores, dtype=np.float32)
    scores[~np.isfinite(scores)] = 0.0
    return scores.tolist()


def _coerce_layer_idx(layer_idx: Any) -> int:
    text = str(layer_idx)
    if text.startswith("layer_"):
        text = text.split("_", 1)[1]
    return int(text)


def _matrix_to_float32(matrix: Any) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)


def _normalise_stage_entries(stage_payload: Any) -> Dict[int, np.ndarray | list[np.ndarray]]:
    if not isinstance(stage_payload, dict):
        return {}
    result: Dict[int, np.ndarray | list[np.ndarray]] = {}
    for layer_idx, raw_value in stage_payload.items():
        try:
            numeric_layer = _coerce_layer_idx(layer_idx)
        except (TypeError, ValueError):
            continue
        if isinstance(raw_value, list) and raw_value and not isinstance(raw_value[0], (int, float, np.number)):
            rows: list[np.ndarray] = []
            for row in raw_value:
                row_np = _matrix_to_float32(row)
                if row_np.ndim == 1:
                    rows.append(row_np)
                elif row_np.ndim == 2:
                    rows.extend(row_np.astype(np.float32))
            if rows:
                result[numeric_layer] = rows
            continue
        matrix_np = _matrix_to_float32(raw_value)
        if matrix_np.ndim == 1:
            result[numeric_layer] = [matrix_np]
        elif matrix_np.ndim == 2:
            result[numeric_layer] = matrix_np
    return result


def normalise_attention_capture(
    *,
    capture: Optional[Dict[str, Any]] = None,
    matrices: Optional[Dict[int, Any]] = None,
) -> Dict[str, Dict[int, np.ndarray | list[np.ndarray]]]:
    stages: Dict[str, Dict[int, np.ndarray | list[np.ndarray]]] = {
        ATTENTION_STAGE_PREFILL: {},
        ATTENTION_STAGE_GENERATION: {},
    }
    if isinstance(capture, dict):
        if any(stage in capture for stage in ATTENTION_CAPTURE_STAGES):
            for stage in ATTENTION_CAPTURE_STAGES:
                stages[stage] = _normalise_stage_entries(capture.get(stage))
        elif isinstance(capture.get("matrices"), dict):
            stages[ATTENTION_STAGE_PREFILL] = _normalise_stage_entries(capture.get("matrices"))
    if matrices:
        stages[ATTENTION_STAGE_PREFILL] = _normalise_stage_entries(matrices)
    return stages


def _stack_generation_rows(
    rows: Any,
    *,
    prompt_len: int,
    output_token_len: int,
) -> Optional[np.ndarray]:
    if rows is None:
        return None
    if isinstance(rows, np.ndarray):
        rows_list = [row.astype(np.float32) for row in rows] if rows.ndim == 2 else [rows.astype(np.float32)]
    else:
        rows_list = []
        for row in rows:
            row_np = _matrix_to_float32(row)
            if row_np.ndim == 1:
                rows_list.append(row_np)
            elif row_np.ndim == 2:
                rows_list.extend(row_np.astype(np.float32))
    if output_token_len <= 0:
        return None
    full_len = prompt_len + output_token_len
    matrix = np.full((output_token_len, full_len), np.nan, dtype=np.float32)
    for row_idx, row_np in enumerate(rows_list[:output_token_len]):
        width = min(row_np.shape[-1], full_len)
        matrix[row_idx, :width] = row_np[:width]
    return matrix


def finalise_attention_stage_matrices(
    *,
    capture: Optional[Dict[str, Any]] = None,
    matrices: Optional[Dict[int, Any]] = None,
    prompt_token_len: Optional[int] = None,
    output_token_len: Optional[int] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    prompt_len = int(prompt_token_len or 0)
    output_len = int(output_token_len or 0)
    raw_stages = normalise_attention_capture(capture=capture, matrices=matrices)
    finalised: Dict[str, Dict[int, np.ndarray]] = {
        ATTENTION_STAGE_PREFILL: {},
        ATTENTION_STAGE_GENERATION: {},
    }
    for layer_idx, matrix in raw_stages[ATTENTION_STAGE_PREFILL].items():
        matrix_np = _matrix_to_float32(matrix)
        if matrix_np.ndim != 2:
            continue
        finalised[ATTENTION_STAGE_PREFILL][int(layer_idx)] = matrix_np
        if prompt_len <= 0:
            prompt_len = int(matrix_np.shape[-1])
    for layer_idx, rows in raw_stages[ATTENTION_STAGE_GENERATION].items():
        generation_matrix = _stack_generation_rows(
            rows,
            prompt_len=prompt_len,
            output_token_len=output_len,
        )
        if generation_matrix is None:
            continue
        finalised[ATTENTION_STAGE_GENERATION][int(layer_idx)] = generation_matrix
    return finalised


def available_attention_layers(stage_matrices: Dict[str, Dict[int, np.ndarray]]) -> list[int]:
    layers = set()
    for stage_payload in stage_matrices.values():
        layers.update(int(layer_idx) for layer_idx in stage_payload.keys())
    return sorted(layers)


def available_attention_stages(stage_matrices: Dict[str, Dict[int, np.ndarray]]) -> list[str]:
    return [stage for stage in ATTENTION_CAPTURE_STAGES if stage_matrices.get(stage)]


def build_combined_attention_matrix(
    *,
    prefill_matrix: Optional[Any],
    generation_matrix: Optional[Any],
) -> Optional[np.ndarray]:
    prefill_np = None if prefill_matrix is None else _matrix_to_float32(prefill_matrix)
    generation_np = None if generation_matrix is None else _matrix_to_float32(generation_matrix)
    if prefill_np is None and generation_np is None:
        return None
    prompt_len = 0
    output_len = 0
    full_len = 0
    if prefill_np is not None and prefill_np.ndim == 2:
        prompt_len = int(prefill_np.shape[0])
        full_len = max(full_len, int(prefill_np.shape[1]))
    else:
        prefill_np = None
    if generation_np is not None and generation_np.ndim == 2:
        output_len = int(generation_np.shape[0])
        full_len = max(full_len, int(generation_np.shape[1]))
        if prompt_len <= 0:
            prompt_len = max(0, full_len - output_len)
    else:
        generation_np = None
    full_len = max(full_len, prompt_len + output_len)
    if full_len <= 0:
        return None
    combined = np.full((full_len, full_len), np.nan, dtype=np.float32)
    if prefill_np is not None:
        rows = min(prompt_len, prefill_np.shape[0])
        cols = min(prompt_len, prefill_np.shape[1], full_len)
        combined[:rows, :cols] = prefill_np[:rows, :cols]
    if generation_np is not None:
        rows = min(output_len, generation_np.shape[0], full_len - prompt_len)
        cols = min(generation_np.shape[1], full_len)
        combined[prompt_len : prompt_len + rows, :cols] = generation_np[:rows, :cols]
    return combined


def get_stage_matrix_for_layer(
    stage_matrices: Dict[str, Dict[int, np.ndarray]],
    *,
    layer_idx: int,
    stage: str,
) -> Optional[np.ndarray]:
    stage_name = (stage or ATTENTION_STAGE_COMBINED).strip().lower()
    if stage_name == ATTENTION_STAGE_COMBINED:
        return build_combined_attention_matrix(
            prefill_matrix=stage_matrices.get(ATTENTION_STAGE_PREFILL, {}).get(int(layer_idx)),
            generation_matrix=stage_matrices.get(ATTENTION_STAGE_GENERATION, {}).get(int(layer_idx)),
        )
    if stage_name not in ATTENTION_CAPTURE_STAGES:
        raise ValueError(
            f"Unsupported attention stage: {stage}. Supported stages: {', '.join(ATTENTION_VIEW_STAGES)}."
        )
    return stage_matrices.get(stage_name, {}).get(int(layer_idx))


def build_stage_shape_index(stage_matrices: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Dict[str, list[int]]]:
    shape_index: Dict[str, Dict[str, list[int]]] = {}
    for stage, stage_payload in stage_matrices.items():
        if not stage_payload:
            continue
        shape_index[stage] = {
            str(int(layer_idx)): list(np.asarray(matrix, dtype=np.float32).shape)
            for layer_idx, matrix in stage_payload.items()
        }
    return shape_index


def build_stage_stats_index(stage_matrices: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats_index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stage, stage_payload in stage_matrices.items():
        if not stage_payload:
            continue
        stats_index[stage] = {
            str(int(layer_idx)): compute_matrix_stats(matrix)
            for layer_idx, matrix in stage_payload.items()
        }
    return stats_index


def normalize_detail_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Attention detail payload must be a dictionary.")
    prompt_token_ids = payload.get("prompt_token_ids")
    prompt_tokens = payload.get("prompt_tokens")
    response_token_ids = payload.get("response_token_ids")
    response_tokens = payload.get("response_tokens")
    if prompt_token_ids is None:
        prompt_token_ids = payload.get("token_ids", [])
    if prompt_tokens is None:
        prompt_tokens = payload.get("tokens", [])
    if response_token_ids is None:
        response_token_ids = []
    if response_tokens is None:
        response_tokens = []
    full_token_ids = payload.get("full_token_ids")
    full_tokens = payload.get("full_tokens")
    if full_token_ids is None:
        full_token_ids = list(prompt_token_ids) + list(response_token_ids)
    if full_tokens is None:
        full_tokens = list(prompt_tokens) + list(response_tokens)

    stage_matrices = finalise_attention_stage_matrices(
        capture=payload.get("capture"),
        matrices=None,
        prompt_token_len=payload.get("input_token_len") or len(prompt_tokens),
        output_token_len=payload.get("output_token_len") or len(response_tokens),
    )
    if not available_attention_layers(stage_matrices) and isinstance(payload.get("available_layers"), list):
        # Legacy payloads do not include capture; their layer info comes from JSON metadata + NPZ.
        stage_matrices = {
            ATTENTION_STAGE_PREFILL: {},
            ATTENTION_STAGE_GENERATION: {},
        }

    payload["prompt_token_ids"] = [int(token_id) for token_id in prompt_token_ids]
    payload["prompt_tokens"] = list(prompt_tokens)
    payload["response_token_ids"] = [int(token_id) for token_id in response_token_ids]
    payload["response_tokens"] = list(response_tokens)
    payload["full_token_ids"] = [int(token_id) for token_id in full_token_ids]
    payload["full_tokens"] = list(full_tokens)
    payload["token_ids"] = list(payload["full_token_ids"])
    payload["tokens"] = list(payload["full_tokens"])
    payload["input_token_len"] = int(payload.get("input_token_len") or len(payload["prompt_tokens"]))
    payload["output_token_len"] = int(payload.get("output_token_len") or len(payload["response_tokens"]))
    if "capture" in payload:
        payload["capture"] = stage_matrices
    payload.setdefault(
        "available_stages",
        available_attention_stages(stage_matrices) or [ATTENTION_STAGE_PREFILL],
    )
    payload.setdefault(
        "stage_matrix_shapes",
        build_stage_shape_index(stage_matrices) if any(stage_matrices.values()) else {},
    )
    payload.setdefault(
        "stage_matrix_stats",
        build_stage_stats_index(stage_matrices) if any(stage_matrices.values()) else {},
    )
    return payload


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
    capture: Optional[Dict[str, Any]] = None,
    matrices: Optional[Dict[int, Any]] = None,
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
    response_token_ids: Optional[list[int]] = None,
    response_token_texts: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist one agent call as structured attention visualization artifacts."""
    stage_matrices = finalise_attention_stage_matrices(
        capture=capture,
        matrices=matrices,
        prompt_token_len=input_token_len,
        output_token_len=output_token_len,
    )
    if not any(stage_matrices.values()):
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
    available_layers = available_attention_layers(stage_matrices)
    available_stages = available_attention_stages(stage_matrices)
    stage_matrix_shapes = build_stage_shape_index(stage_matrices)
    stage_matrix_stats = build_stage_stats_index(stage_matrices)

    for stage, stage_payload in stage_matrices.items():
        for layer_idx, matrix in stage_payload.items():
            matrix_np = np.asarray(matrix, dtype=np.float32)
            if matrix_np.ndim != 2:
                logger.warning(
                    "Skipping invalid attention matrix for request={} agent={} stage={} layer={} with shape {}.",
                    request_uid,
                    agent_id,
                    stage,
                    layer_idx,
                    tuple(matrix_np.shape),
                )
                continue
            serialisable_matrices[f"{stage}_layer_{int(layer_idx)}"] = matrix_np

    if not serialisable_matrices:
        logger.warning(
            "Attention visualization export skipped because no 2D matrices were serialisable for request={} agent={}.",
            request_uid,
            agent_id,
        )
        return None

    np.savez_compressed(npz_path, **serialisable_matrices)

    prompt_token_ids = [int(token_id) for token_id in input_token_ids]
    prompt_tokens = list(input_token_texts)
    response_token_ids = [int(token_id) for token_id in (response_token_ids or [])]
    response_tokens = list(response_token_texts or [])
    full_token_ids = prompt_token_ids + response_token_ids
    full_tokens = prompt_tokens + response_tokens

    detail_payload = {
        "run_tag": run_tag,
        "request_uid": request_uid,
        "round_index": round_index,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_role": agent_role,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": prompt_tokens,
        "response_token_ids": response_token_ids,
        "response_tokens": response_tokens,
        "full_token_ids": full_token_ids,
        "full_tokens": full_tokens,
        "token_ids": full_token_ids,
        "tokens": full_tokens,
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "available_layers": available_layers,
        "available_stages": available_stages,
        "stage_matrix_shapes": stage_matrix_shapes,
        "stage_matrix_stats": stage_matrix_stats,
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
        "available_stages": available_stages,
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
        "available_stages": available_stages,
    }
    _log_artifact(artifacts)
    return artifacts
