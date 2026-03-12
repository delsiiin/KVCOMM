from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class GenerationResult:
    """Container describing a single model generation outcome."""

    text: str
    mode: str
    ttft: float
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestMetricsRecorder:
    """Tracks per-request agent outputs, default-mode rates, and mode-specific TTFT."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_calls: int = 0
        self._total_default_calls: int = 0
        self._ttft_stats: Dict[str, Dict[str, float]] = {}
        self._agent_events: List[Dict[str, Any]] = []
        self._tool_events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all cached request-level and per-agent metrics."""
        with self._lock:
            self._requests.clear()
            self._total_calls = 0
            self._total_default_calls = 0
            self._ttft_stats.clear()
            self._agent_events.clear()
            self._tool_events.clear()

    def start_request(
        self,
        *,
        request_uid: str,
        batch_index: Optional[int],
        task: Optional[str],
        execution_mode: str,
    ) -> None:
        """Initialise bookkeeping for a new request."""
        with self._lock:
            self._requests[request_uid] = {
                "batch_index": batch_index,
                "task": task,
                "execution_mode": execution_mode,
                "agents": [],
                "default_count": 0,
                "total_count": 0,
            }

    def record_agent_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        generation: GenerationResult | None,
        round_index: Optional[int] = None,
    ) -> None:
        """Log an agent's generation output and update reuse statistics."""
        if generation is None:
            return

        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "default_count": 0,
                    "total_count": 0,
                },
            )
            metadata = generation.metadata or {}
            round_value = metadata.get("round_index", round_index)
            input_char_len = self._normalise_length(metadata.get("input_char_len"))
            output_char_len = self._normalise_length(metadata.get("output_char_len"))
            input_token_len = self._normalise_length(metadata.get("input_token_len"))
            output_token_len = self._normalise_length(metadata.get("output_token_len"))
            if output_char_len is None:
                output_char_len = len(generation.text) if generation.text is not None else 0

            agent_record = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "text": generation.text,
                "round_index": round_value,
            }
            if metadata:
                agent_record["metadata"] = metadata
            agents_list = request_entry["agents"]
            replaced = False
            for idx, existing in enumerate(agents_list):
                if existing["agent_id"] == agent_id:
                    agents_list[idx] = agent_record
                    replaced = True
                    break
            if not replaced:
                agents_list.append(agent_record)


            request_entry["total_count"] = len(agents_list)
            request_entry["default_count"] = sum(
                1 for entry in agents_list if entry.get("mode") == "default"
            )


            stats = self._ttft_stats.setdefault(
                generation.mode, {"sum": 0.0, "count": 0.0}
            )
            stats["sum"] += generation.ttft
            stats["count"] += 1
            avg_ttft = stats["sum"] / stats["count"] if stats["count"] else 0.0

            ttft_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "mode_avg_ttft": avg_ttft,
                "round_index": round_value,
            }
            preprocess_latency = metadata.get("preprocess_latency")
            if preprocess_latency is not None:
                ttft_payload["preprocess_latency"] = preprocess_latency
            generation_ttft = metadata.get("generation_ttft")
            if generation_ttft is not None:
                ttft_payload["generation_ttft"] = generation_ttft
            logger.opt(colors=True).info(
                "<cyan>[TTFT:{mode}]</cyan> {}",
                json.dumps(ttft_payload, ensure_ascii=False),
                mode=generation.mode,
            )

            output_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "text": generation.text,
                "round_index": round_value,
                "input_char_len": input_char_len,
                "output_char_len": output_char_len,
                "input_token_len": input_token_len,
                "output_token_len": output_token_len,
            }
            logger.opt(colors=True).info(
                "<green>[AGENT OUTPUT]</green> {}",
                json.dumps(output_payload, ensure_ascii=False),
            )

            self._agent_events.append(
                {
                    "request_uid": request_uid,
                    "batch_index": request_entry.get("batch_index"),
                    "task": request_entry.get("task"),
                    "execution_mode": request_entry.get("execution_mode"),
                    "round_index": round_value,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "mode": generation.mode,
                    "ttft": generation.ttft,
                    "input_char_len": input_char_len,
                    "output_char_len": output_char_len,
                    "input_token_len": input_token_len,
                    "output_token_len": output_token_len,
                }
            )

    def record_tool_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        tool_name: str,
        tool_output_text: str,
        tool_output_char_len: Optional[int] = None,
        tool_output_token_len: Optional[int] = None,
        round_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one tool-call output event for later aggregation/export."""
        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "default_count": 0,
                    "total_count": 0,
                },
            )
            normalised_text = "" if tool_output_text is None else str(tool_output_text)
            payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "execution_mode": request_entry.get("execution_mode"),
                "round_index": round_index,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "tool_name": tool_name,
                "tool_output_text": normalised_text,
                "tool_output_char_len": self._normalise_length(tool_output_char_len)
                if tool_output_char_len is not None
                else len(normalised_text),
                "tool_output_token_len": self._normalise_length(tool_output_token_len),
            }
            if metadata:
                payload["metadata"] = metadata
            logger.opt(colors=True).info(
                "<green>[TOOL OUTPUT]</green> {}",
                json.dumps(payload, ensure_ascii=False),
            )
            self._tool_events.append(payload)

    def finalize_request(self, request_uid: str) -> Optional[float]:
        """Compute and log per-request default-mode statistics."""
        with self._lock:
            request_entry = self._requests.pop(request_uid, None)
            if request_entry is None:
                return None

            default_count = request_entry.get("default_count", 0)
            total = request_entry.get("total_count", 0)
            default_rate = (default_count / total) if total else 0.0

            payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "execution_mode": request_entry.get("execution_mode"),
                "default_rate": default_rate,
                "default_count": default_count,
                "total_agents": total,
            }
            logger.opt(colors=True).info(
                "<magenta>[REQUEST MODE]</magenta> {}",
                json.dumps(payload, ensure_ascii=False),
            )

            self._total_calls += total
            self._total_default_calls += default_count
            return default_rate

    def log_cumulative(self, *, batch_index: Optional[int]) -> float:
        """Log the cumulative average default-mode rate across all requests."""
        with self._lock:
            if self._total_calls == 0:
                cumulative = 0.0
            else:
                cumulative = self._total_default_calls / self._total_calls

            payload = {
                "batch_index": batch_index,
                "cumulative_default_rate": cumulative,
                "default_calls": self._total_default_calls,
                "total_agent_calls": self._total_calls,
            }
            logger.opt(colors=True).info(
                "<yellow>[CUMULATIVE MODE]</yellow> {}",
                json.dumps(payload, ensure_ascii=False),
            )
            return cumulative

    @staticmethod
    def _normalise_length(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    @staticmethod
    def _describe(values: List[int]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0,
                "min": 0,
                "p50": 0,
                "p90": 0,
                "max": 0,
                "mean": 0,
            }
        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            if n == 1:
                return float(sorted_values[0])
            pos = (n - 1) * p
            lower = int(math.floor(pos))
            upper = int(math.ceil(pos))
            if lower == upper:
                return float(sorted_values[lower])
            weight = pos - lower
            return float(
                sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
            )

        return {
            "count": n,
            "min": int(sorted_values[0]),
            "p50": round(percentile(0.5), 2),
            "p90": round(percentile(0.9), 2),
            "max": int(sorted_values[-1]),
            "mean": round(sum(sorted_values) / n, 2),
        }

    @staticmethod
    def _choose_bin_step(max_value: int, target_bins: int = 14) -> int:
        if max_value <= 0:
            return 1
        rough = max(1, int(math.ceil(max_value / target_bins)))
        magnitude = 10 ** int(math.floor(math.log10(rough)))
        for multiplier in (1, 2, 5, 10):
            step = multiplier * magnitude
            if step >= rough:
                return step
        return rough

    @staticmethod
    def _agent_sort_key(agent_id: str) -> tuple[int, Any]:
        text = str(agent_id)
        return (0, int(text)) if text.isdigit() else (1, text)

    def _plot_hist_by_agent(
        self,
        *,
        events: List[Dict[str, Any]],
        value_key: str,
        value_name: str,
        out_path: Path,
        run_tag: str,
        unit: str,
    ) -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Matplotlib unavailable, skip plotting {}: {}", value_name, exc)
            return None

        grouped: Dict[str, List[int]] = {}
        role_by_agent: Dict[str, set[str]] = {}
        for event in events:
            value = self._normalise_length(event.get(value_key))
            if value is None:
                continue
            agent_id = str(event.get("agent_id"))
            grouped.setdefault(agent_id, []).append(value)
            role = str(event.get("agent_role", "")).strip()
            if role:
                role_by_agent.setdefault(agent_id, set()).add(role)

        if not grouped:
            logger.warning("No valid values for {}. Plot skipped.", value_name)
            return None

        agent_ids = sorted(grouped.keys(), key=self._agent_sort_key)
        max_value = max(max(values) for values in grouped.values())
        step = self._choose_bin_step(max_value)
        num_bins = max(1, int(math.ceil((max_value + 1) / step)))
        bin_labels = [f"{i * step}-{(i + 1) * step - 1}" for i in range(num_bins)]

        fig_width = max(12.0, min(28.0, 0.8 * num_bins))
        fig_height = max(4.0, 2.6 * len(agent_ids))
        fig, axes = plt.subplots(
            len(agent_ids),
            1,
            figsize=(fig_width, fig_height),
            sharex=True,
        )
        if len(agent_ids) == 1:
            axes = [axes]

        for ax, agent_id in zip(axes, agent_ids):
            counts = [0] * num_bins
            for value in grouped[agent_id]:
                index = min(value // step, num_bins - 1)
                counts[index] += 1
            ax.bar(range(num_bins), counts, color="#3a6ea5", width=0.85)
            role_values = sorted(role_by_agent.get(agent_id, set()))
            role_label = role_values[0] if len(role_values) == 1 else "|".join(role_values)
            if not role_label:
                role_label = "unknown-role"
            ax.set_ylabel(f"id {agent_id}\n{role_label}\ncount")
            ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)

        tick_step = max(1, num_bins // 18)
        ticks = list(range(0, num_bins, tick_step))
        if ticks[-1] != num_bins - 1:
            ticks.append(num_bins - 1)
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels([bin_labels[idx] for idx in ticks], rotation=35, ha="right")
        axes[-1].set_xlabel(f"{value_name} ({unit}, bin={step})")
        fig.suptitle(f"{run_tag} | {value_name} distribution by agent_id (with role)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def export_tool_length_artifacts(
        self,
        *,
        output_dir: str | Path,
        run_tag: str,
        plot_hist: bool = True,
    ) -> Dict[str, str]:
        """Dump per-agent tool-output events and histogram plot to disk."""
        with self._lock:
            events = list(self._tool_events)
        if not events:
            logger.warning("No tool events captured for {}. Skip exporting tool length artifacts.", run_tag)
            return {}

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        events_file = output_path / f"{run_tag}_tool_output_events.json"
        with events_file.open("w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)

        per_agent: Dict[str, Dict[str, Any]] = {}
        for event in events:
            agent_id = str(event.get("agent_id"))
            group = per_agent.setdefault(
                agent_id,
                {
                    "agent_name": event.get("agent_name"),
                    "agent_role": event.get("agent_role"),
                    "round_indices": set(),
                    "tool_names": set(),
                    "tool_call_count": 0,
                    "tool_output_values": [],
                },
            )
            if (not group.get("agent_name")) and event.get("agent_name"):
                group["agent_name"] = event.get("agent_name")
            if (not group.get("agent_role")) and event.get("agent_role"):
                group["agent_role"] = event.get("agent_role")
            round_index = event.get("round_index")
            if round_index is not None:
                group["round_indices"].add(round_index)
            tool_name = str(event.get("tool_name", "")).strip()
            if tool_name:
                group["tool_names"].add(tool_name)
            group["tool_call_count"] += 1
            tool_output_token_len = self._normalise_length(event.get("tool_output_token_len"))
            if tool_output_token_len is not None:
                group["tool_output_values"].append(tool_output_token_len)

        summary = {
            "run_tag": run_tag,
            "unit": "tokens",
            "tool_output_length_field": "tool_output_token_len",
            "num_events": len(events),
            "agent_id_role_map": {},
            "agents": {},
        }
        for agent_id, data in sorted(per_agent.items(), key=lambda item: self._agent_sort_key(item[0])):
            tool_output_values = data.pop("tool_output_values")
            round_indices = sorted(data.pop("round_indices"))
            tool_names = sorted(data.pop("tool_names"))
            tool_call_count = data.pop("tool_call_count")
            summary["agent_id_role_map"][agent_id] = data.get("agent_role")
            summary["agents"][agent_id] = {
                **data,
                "round_indices": round_indices,
                "round_count": len(round_indices),
                "tool_names": tool_names,
                "tool_call_count": tool_call_count,
                "tool_output_stats": self._describe(tool_output_values),
            }

        summary_file = output_path / f"{run_tag}_tool_output_summary.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        output_plot = None
        if plot_hist:
            output_plot = self._plot_hist_by_agent(
                events=events,
                value_key="tool_output_token_len",
                value_name="Tool output length",
                out_path=output_path / f"{run_tag}_tool_output_length_hist_by_agent.png",
                run_tag=run_tag,
                unit="tokens",
            )

        artifacts = {
            "events_json": str(events_file),
            "summary_json": str(summary_file),
            "plot_hist": str(plot_hist),
        }
        if output_plot is not None:
            artifacts["output_hist_png"] = str(output_plot)
        logger.opt(colors=True).info(
            "<blue>[TOOL LENGTH ARTIFACTS]</blue> {}",
            json.dumps(artifacts, ensure_ascii=False),
        )
        return artifacts

    def export_agent_length_artifacts(
        self,
        *,
        output_dir: str | Path,
        run_tag: str,
        plot_hist: bool = True,
    ) -> Dict[str, str]:
        """Dump per-agent length events and histogram plots to disk."""
        with self._lock:
            events = list(self._agent_events)
        if not events:
            logger.warning("No agent events captured for {}. Skip exporting length artifacts.", run_tag)
            return {}

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        all_have_token_lengths = all(
            self._normalise_length(event.get("input_token_len")) is not None
            and self._normalise_length(event.get("output_token_len")) is not None
            for event in events
        )
        if all_have_token_lengths:
            input_key = "input_token_len"
            output_key = "output_token_len"
            unit = "tokens"
        else:
            input_key = "input_char_len"
            output_key = "output_char_len"
            unit = "chars"

        events_file = output_path / f"{run_tag}_agent_length_events.json"
        with events_file.open("w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)

        per_agent: Dict[str, Dict[str, Any]] = {}
        for event in events:
            agent_id = str(event.get("agent_id"))
            group = per_agent.setdefault(
                agent_id,
                {
                    "agent_name": event.get("agent_name"),
                    "agent_role": event.get("agent_role"),
                    "round_indices": set(),
                    "input_values": [],
                    "output_values": [],
                },
            )
            if (not group.get("agent_name")) and event.get("agent_name"):
                group["agent_name"] = event.get("agent_name")
            if (not group.get("agent_role")) and event.get("agent_role"):
                group["agent_role"] = event.get("agent_role")
            round_index = event.get("round_index")
            if round_index is not None:
                group["round_indices"].add(round_index)
            input_value = self._normalise_length(event.get(input_key))
            output_value = self._normalise_length(event.get(output_key))
            if input_value is not None:
                group["input_values"].append(input_value)
            if output_value is not None:
                group["output_values"].append(output_value)

        summary = {
            "run_tag": run_tag,
            "unit": unit,
            "input_length_field": input_key,
            "output_length_field": output_key,
            "num_events": len(events),
            "agent_id_role_map": {},
            "agents": {},
        }
        for agent_id, data in sorted(per_agent.items(), key=lambda item: self._agent_sort_key(item[0])):
            input_values = data.pop("input_values")
            output_values = data.pop("output_values")
            round_indices = sorted(data.pop("round_indices"))
            summary["agent_id_role_map"][agent_id] = data.get("agent_role")
            summary["agents"][agent_id] = {
                **data,
                "round_indices": round_indices,
                "round_count": len(round_indices),
                "input_stats": self._describe(input_values),
                "output_stats": self._describe(output_values),
            }

        summary_file = output_path / f"{run_tag}_agent_length_summary.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        input_plot = None
        output_plot = None
        if plot_hist:
            input_plot = self._plot_hist_by_agent(
                events=events,
                value_key=input_key,
                value_name="Input length",
                out_path=output_path / f"{run_tag}_input_length_hist_by_agent.png",
                run_tag=run_tag,
                unit=unit,
            )
            output_plot = self._plot_hist_by_agent(
                events=events,
                value_key=output_key,
                value_name="Output length",
                out_path=output_path / f"{run_tag}_output_length_hist_by_agent.png",
                run_tag=run_tag,
                unit=unit,
            )

        artifacts = {
            "events_json": str(events_file),
            "summary_json": str(summary_file),
        }
        artifacts["plot_hist"] = str(plot_hist)
        if input_plot is not None:
            artifacts["input_hist_png"] = str(input_plot)
        if output_plot is not None:
            artifacts["output_hist_png"] = str(output_plot)
        logger.opt(colors=True).info(
            "<blue>[LENGTH ARTIFACTS]</blue> {}",
            json.dumps(artifacts, ensure_ascii=False),
        )
        return artifacts


metrics_recorder = RequestMetricsRecorder()
