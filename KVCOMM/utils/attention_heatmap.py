from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


def build_heatmap_tag(
    attn_heatmap_mode: bool,
    attn_heatmap_layer: Optional[int],
) -> str:
    """Return a stable tag segment for attention heatmap artifacts."""
    if not attn_heatmap_mode:
        return "no-heatmap"
    if attn_heatmap_layer is None:
        return "heatmap-layer-unknown"
    return f"heatmap-l{int(attn_heatmap_layer)}"


def sanitize_filename_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "unknown"


def export_attention_heatmap(
    *,
    matrix: Any,
    output_dir: str | Path,
    run_tag: str,
    request_uid: Optional[str],
    round_index: Optional[int],
    agent_id: Optional[str],
    agent_name: Optional[str],
    agent_role: Optional[str],
    layer_idx: int,
    input_token_len: Optional[int] = None,
    output_token_len: Optional[int] = None,
) -> Optional[Dict[str, str]]:
    """Persist one attention heatmap PNG plus JSON metadata sidecar."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Matplotlib unavailable, skip attention heatmap export: {}", exc)
        return None

    matrix_np = np.asarray(matrix, dtype=np.float32)
    if matrix_np.ndim != 2:
        logger.warning(
            "Attention heatmap expects a 2D matrix, got shape {}. Skip export.",
            tuple(matrix_np.shape),
        )
        return None

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    safe_request_uid = sanitize_filename_component(request_uid or "no-request")
    safe_agent_id = sanitize_filename_component(agent_id or "unknown-agent")
    safe_agent_role = sanitize_filename_component(agent_role or "unknown-role")
    safe_agent_name = sanitize_filename_component(agent_name or "unknown-name")
    safe_round = "na" if round_index is None else str(int(round_index))
    stem = (
        f"{sanitize_filename_component(run_tag)}"
        f"_req-{safe_request_uid}"
        f"_round-{safe_round}"
        f"_agent-{safe_agent_id}"
        f"_role-{safe_agent_role}"
        f"_layer-{int(layer_idx)}"
    )

    png_path = output_path / f"{stem}.png"
    json_path = output_path / f"{stem}.json"

    fig_width = max(8.0, min(22.0, matrix_np.shape[1] / 24.0))
    fig_height = max(6.0, min(18.0, matrix_np.shape[0] / 24.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix_np, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=0.01)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(
        "\n".join(
            [
                run_tag,
                (
                    f"request={request_uid or 'n/a'} | round={safe_round} | "
                    f"agent_id={agent_id or 'n/a'} | role={agent_role or 'n/a'} | "
                    f"layer={int(layer_idx)}"
                ),
            ]
        )
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "run_tag": run_tag,
        "request_uid": request_uid,
        "round_index": round_index,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_role": agent_role,
        "layer_idx": int(layer_idx),
        "shape": [int(matrix_np.shape[0]), int(matrix_np.shape[1])],
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "png_path": str(png_path),
        "agent_name_sanitized": safe_agent_name,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    artifacts = {
        "png_path": str(png_path),
        "json_path": str(json_path),
    }
    logger.opt(colors=True).info(
        "<blue>[ATTN HEATMAP]</blue> {}",
        json.dumps(artifacts, ensure_ascii=False),
    )
    return artifacts
