import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from KVCOMM.utils.attention_heatmap import (
    compute_token_scores,
    sanitize_filename_component,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


class AttentionSessionStore:
    def __init__(self, session_root: str | Path):
        self.session_root = Path(session_root).expanduser()

    def _run_dirs(self) -> List[Path]:
        if not self.session_root.exists():
            return []
        return sorted(
            [path for path in self.session_root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        )

    def _index_path(self, run_dir: Path) -> Path:
        return run_dir / "session_index.json"

    def _load_run_payload(self, run_dir: Path) -> Dict[str, Any]:
        index_path = self._index_path(run_dir)
        if not index_path.exists():
            return {"run_tag": run_dir.name, "calls": []}
        payload = _load_json(index_path)
        payload.setdefault("run_tag", run_dir.name)
        payload.setdefault("calls", [])
        return payload

    def _find_run_dir(self, run_tag: str) -> Optional[Path]:
        safe_run_tag = sanitize_filename_component(run_tag)
        direct = self.session_root / safe_run_tag
        if direct.exists():
            return direct
        for run_dir in self._run_dirs():
            payload = self._load_run_payload(run_dir)
            if payload.get("run_tag") == run_tag:
                return run_dir
        return None

    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for run_dir in self._run_dirs():
            payload = self._load_run_payload(run_dir)
            calls = payload.get("calls", [])
            request_ids = {call.get("request_uid") for call in calls if call.get("request_uid")}
            runs.append(
                {
                    "run_tag": payload.get("run_tag", run_dir.name),
                    "request_count": len(request_ids),
                    "call_count": len(calls),
                }
            )
        return runs

    def get_run_index(self, run_tag: str) -> Dict[str, Any]:
        run_dir = self._find_run_dir(run_tag)
        if run_dir is None:
            raise FileNotFoundError(f"Run not found: {run_tag}")
        payload = self._load_run_payload(run_dir)
        tree: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for call in payload.get("calls", []):
            request_uid = str(call.get("request_uid") or "unknown-request")
            round_index = int(call.get("round_index") or 0)
            tree.setdefault(request_uid, {}).setdefault(round_index, []).append(
                {
                    "agent_id": call.get("agent_id"),
                    "agent_name": call.get("agent_name"),
                    "agent_role": call.get("agent_role"),
                    "available_layers": call.get("available_layers", []),
                }
            )
        requests: List[Dict[str, Any]] = []
        for request_uid in sorted(tree.keys()):
            rounds: List[Dict[str, Any]] = []
            for round_index in sorted(tree[request_uid].keys()):
                agents = sorted(
                    tree[request_uid][round_index],
                    key=lambda item: (
                        str(item.get("agent_name") or ""),
                        str(item.get("agent_id") or ""),
                    ),
                )
                rounds.append({"round_index": round_index, "agents": agents})
            requests.append({"request_uid": request_uid, "rounds": rounds})
        return {"run_tag": payload.get("run_tag", run_dir.name), "requests": requests}

    def _find_call(
        self,
        *,
        run_tag: str,
        request_uid: str,
        agent_id: str,
        round_index: Optional[int],
    ) -> tuple[Path, Dict[str, Any]]:
        run_dir = self._find_run_dir(run_tag)
        if run_dir is None:
            raise FileNotFoundError(f"Run not found: {run_tag}")
        payload = self._load_run_payload(run_dir)
        matches = [
            call
            for call in payload.get("calls", [])
            if call.get("request_uid") == request_uid and call.get("agent_id") == agent_id
        ]
        if round_index is not None:
            matches = [call for call in matches if int(call.get("round_index") or 0) == int(round_index)]
        if not matches:
            raise FileNotFoundError(
                f"Agent call not found: run={run_tag} request={request_uid} round={round_index} agent={agent_id}"
            )
        if len(matches) > 1:
            raise ValueError("round_index is required because multiple rounds exist for this agent.")
        return run_dir, matches[0]

    def get_agent_detail(
        self,
        *,
        run_tag: str,
        request_uid: str,
        agent_id: str,
        round_index: Optional[int],
    ) -> Dict[str, Any]:
        run_dir, call = self._find_call(
            run_tag=run_tag,
            request_uid=request_uid,
            agent_id=agent_id,
            round_index=round_index,
        )
        detail_path = run_dir / str(call["json_path"])
        payload = _load_json(detail_path)
        payload["run_tag"] = run_tag
        return payload

    def get_layer_payload(
        self,
        *,
        run_tag: str,
        request_uid: str,
        agent_id: str,
        round_index: Optional[int],
        layer_idx: int,
    ) -> Dict[str, Any]:
        run_dir, call = self._find_call(
            run_tag=run_tag,
            request_uid=request_uid,
            agent_id=agent_id,
            round_index=round_index,
        )
        detail_path = run_dir / str(call["json_path"])
        detail_payload = _load_json(detail_path)
        npz_path = run_dir / str(call["npz_path"])
        with np.load(npz_path) as matrices:
            key = f"layer_{int(layer_idx)}"
            if key not in matrices:
                raise FileNotFoundError(
                    f"Layer {layer_idx} not found for run={run_tag} request={request_uid} agent={agent_id}"
                )
            matrix = matrices[key].astype(np.float32)
        token_scores = compute_token_scores(matrix)
        return {
            "run_tag": run_tag,
            "request_uid": request_uid,
            "round_index": detail_payload.get("round_index"),
            "agent_id": detail_payload.get("agent_id"),
            "agent_name": detail_payload.get("agent_name"),
            "agent_role": detail_payload.get("agent_role"),
            "layer_idx": int(layer_idx),
            "available_layers": detail_payload.get("available_layers", []),
            "matrix_shape": list(matrix.shape),
            "matrix_stats": (detail_payload.get("matrix_stats") or {}).get(str(int(layer_idx)), {}),
            "tokens": detail_payload.get("tokens", []),
            "token_ids": detail_payload.get("token_ids", []),
            "token_scores": token_scores,
            "matrix": matrix.tolist(),
        }


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KVCOMM Attention Viewer</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: rgba(255, 250, 241, 0.88);
      --panel-strong: #fffaf1;
      --ink: #1d2730;
      --muted: #5d6a72;
      --line: rgba(29, 39, 48, 0.12);
      --accent: #155d7a;
      --accent-soft: #d5ebf4;
      --shadow: 0 18px 40px rgba(33, 49, 58, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(34, 125, 164, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(205, 144, 88, 0.18), transparent 24%),
        linear-gradient(180deg, #faf6ef 0%, #f1ece4 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 22px 24px;
      margin-bottom: 18px;
      backdrop-filter: blur(10px);
    }
    .hero h1 {
      margin: 0;
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: 32px;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }
    .hero p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 900px;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(10px);
    }
    .controls {
      position: sticky;
      top: 18px;
      align-self: start;
    }
    .section-title {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .field {
      margin-bottom: 14px;
    }
    .field label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .field select,
    .field input[type="number"],
    .field input[type="range"] {
      width: 100%;
    }
    select,
    input[type="number"] {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      color: var(--ink);
      font: inherit;
    }
    input[type="range"] {
      accent-color: var(--accent);
    }
    .button-row {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 9px 14px;
      cursor: pointer;
      font: inherit;
    }
    button.secondary {
      background: var(--accent-soft);
      color: var(--accent);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .content {
      display: grid;
      gap: 18px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .meta-item {
      background: rgba(255, 255, 255, 0.6);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }
    .meta-item strong {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .heatmap-wrap {
      overflow: auto;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.65);
      padding: 10px;
    }
    canvas {
      display: block;
      image-rendering: pixelated;
      max-width: 100%;
      border-radius: 14px;
      background: white;
    }
    .prompt-tokens {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      padding: 16px;
      line-height: 1.85;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 220px;
    }
    .token {
      border-radius: 6px;
      padding: 1px 0;
      transition: background-color 120ms ease;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.6;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .status {
      color: var(--muted);
      font-size: 14px;
      min-height: 22px;
    }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .controls { position: static; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>KVCOMM Attention Viewer</h1>
      <p>Browse structured prefill attention captures by run, request, round, agent, and layer. The heatmap color range and prompt token shading update live on the page.</p>
    </section>
    <div class="layout">
      <aside class="card controls">
        <h2 class="section-title">Browse</h2>
        <div class="field">
          <label for="runSelect">Run</label>
          <select id="runSelect"></select>
        </div>
        <div class="field">
          <label for="requestSelect">Request</label>
          <select id="requestSelect"></select>
        </div>
        <div class="field">
          <label for="roundSelect">Round</label>
          <select id="roundSelect"></select>
        </div>
        <div class="field">
          <label for="agentSelect">Agent</label>
          <select id="agentSelect"></select>
        </div>
        <h2 class="section-title">Layer</h2>
        <div class="field">
          <label id="layerLabel" for="layerSlider">Layer</label>
          <input id="layerSlider" type="range" min="0" max="0" step="1" value="0" />
          <div class="button-row">
            <button id="prevLayerBtn" class="secondary">Prev</button>
            <button id="nextLayerBtn" class="secondary">Next</button>
          </div>
        </div>
        <h2 class="section-title">Color Range</h2>
        <div class="field">
          <label for="vminInput">vmin</label>
          <input id="vminInput" type="number" step="any" />
          <input id="vminSlider" type="range" min="0" max="1" step="0.001" value="0" />
        </div>
        <div class="field">
          <label for="vmaxInput">vmax</label>
          <input id="vmaxInput" type="number" step="any" />
          <input id="vmaxSlider" type="range" min="0" max="1" step="0.001" value="1" />
          <div class="button-row">
            <button id="resetRangeBtn">Reset</button>
            <button id="robustRangeBtn" class="secondary">Use p1-p99</button>
          </div>
        </div>
        <div id="status" class="status"></div>
      </aside>
      <main class="content">
        <section class="card">
          <h2 class="section-title">Agent Metadata</h2>
          <div class="meta-grid" id="metaGrid"></div>
        </section>
        <section class="card">
          <h2 class="section-title">Attention Heatmap</h2>
          <div class="heatmap-wrap">
            <canvas id="heatmapCanvas"></canvas>
          </div>
          <div class="stats-grid" id="statsGrid"></div>
        </section>
        <section class="card">
          <h2 class="section-title">Prompt Tokens</h2>
          <div id="promptTokens" class="prompt-tokens"></div>
        </section>
        <section class="card">
          <h2 class="section-title">Response</h2>
          <pre id="responseText"></pre>
        </section>
      </main>
    </div>
  </div>
  <script>
    const state = {
      runs: [],
      runData: null,
      agentDetail: null,
      layerData: null,
      selectedRun: null,
      selectedRequest: null,
      selectedRound: null,
      selectedAgent: null,
      selectedLayerIndex: 0,
      vmin: 0,
      vmax: 1,
      absMin: 0,
      absMax: 1,
      robustMin: 0,
      robustMax: 1,
    };

    const runSelect = document.getElementById("runSelect");
    const requestSelect = document.getElementById("requestSelect");
    const roundSelect = document.getElementById("roundSelect");
    const agentSelect = document.getElementById("agentSelect");
    const layerSlider = document.getElementById("layerSlider");
    const layerLabel = document.getElementById("layerLabel");
    const prevLayerBtn = document.getElementById("prevLayerBtn");
    const nextLayerBtn = document.getElementById("nextLayerBtn");
    const vminInput = document.getElementById("vminInput");
    const vmaxInput = document.getElementById("vmaxInput");
    const vminSlider = document.getElementById("vminSlider");
    const vmaxSlider = document.getElementById("vmaxSlider");
    const resetRangeBtn = document.getElementById("resetRangeBtn");
    const robustRangeBtn = document.getElementById("robustRangeBtn");
    const heatmapCanvas = document.getElementById("heatmapCanvas");
    const metaGrid = document.getElementById("metaGrid");
    const statsGrid = document.getElementById("statsGrid");
    const promptTokens = document.getElementById("promptTokens");
    const responseText = document.getElementById("responseText");
    const status = document.getElementById("status");

    function setStatus(message) {
      status.textContent = message || "";
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ("Request failed: " + response.status));
      }
      return response.json();
    }

    function setSelectOptions(select, items, valueKey, labelBuilder) {
      select.innerHTML = "";
      items.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item[valueKey]);
        option.textContent = labelBuilder(item);
        select.appendChild(option);
      });
    }

    function currentRequestData() {
      return (state.runData?.requests || []).find((item) => item.request_uid === state.selectedRequest) || null;
    }

    function currentRoundData() {
      const request = currentRequestData();
      return (request?.rounds || []).find((item) => String(item.round_index) === String(state.selectedRound)) || null;
    }

    function currentAgentSummary() {
      const round = currentRoundData();
      return (round?.agents || []).find((item) => item.agent_id === state.selectedAgent) || null;
    }

    function clamp(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function colorForValue(value, vmin, vmax) {
      const span = Math.max(vmax - vmin, 1e-12);
      const t = clamp((value - vmin) / span, 0, 1);
      const r = Math.round(250 - 190 * t);
      const g = Math.round(248 - 148 * t);
      const b = Math.round(242 - 32 * t);
      return [r, g, b];
    }

    function applyRangeToControls(minValue, maxValue, currentMin, currentMax) {
      const safeMin = Number.isFinite(minValue) ? minValue : 0;
      const safeMax = Number.isFinite(maxValue) && maxValue > safeMin ? maxValue : safeMin + 1e-6;
      const step = Math.max((safeMax - safeMin) / 500, 1e-9);
      [vminSlider, vmaxSlider].forEach((slider) => {
        slider.min = String(safeMin);
        slider.max = String(safeMax);
        slider.step = String(step);
      });
      state.absMin = safeMin;
      state.absMax = safeMax;
      state.vmin = clamp(currentMin, safeMin, safeMax);
      state.vmax = clamp(currentMax, state.vmin, safeMax);
      vminInput.value = String(state.vmin);
      vmaxInput.value = String(state.vmax);
      vminSlider.value = String(state.vmin);
      vmaxSlider.value = String(state.vmax);
    }

    function updateRangeFromInputs(source) {
      let nextMin = parseFloat(vminInput.value);
      let nextMax = parseFloat(vmaxInput.value);
      if (source === "slider") {
        nextMin = parseFloat(vminSlider.value);
        nextMax = parseFloat(vmaxSlider.value);
      }
      nextMin = Number.isFinite(nextMin) ? nextMin : state.absMin;
      nextMax = Number.isFinite(nextMax) ? nextMax : state.absMax;
      if (nextMin > nextMax) {
        if (source === "vmin") nextMax = nextMin;
        else nextMin = nextMax;
      }
      state.vmin = clamp(nextMin, state.absMin, state.absMax);
      state.vmax = clamp(nextMax, state.vmin, state.absMax);
      vminInput.value = String(state.vmin);
      vmaxInput.value = String(state.vmax);
      vminSlider.value = String(state.vmin);
      vmaxSlider.value = String(state.vmax);
      renderLayerVisuals();
    }

    function renderMetadata() {
      const detail = state.agentDetail;
      metaGrid.innerHTML = "";
      if (!detail) return;
      const items = [
        ["Run", state.selectedRun],
        ["Request", detail.request_uid],
        ["Round", detail.round_index],
        ["Agent", detail.agent_id],
        ["Name", detail.agent_name || "n/a"],
        ["Role", detail.agent_role || "n/a"],
        ["Prompt Tokens", detail.input_token_len ?? detail.tokens?.length ?? 0],
        ["Response Tokens", detail.output_token_len ?? 0],
      ];
      items.forEach(([label, value]) => {
        const box = document.createElement("div");
        box.className = "meta-item";
        box.innerHTML = "<strong>" + label + "</strong><span>" + String(value ?? "n/a") + "</span>";
        metaGrid.appendChild(box);
      });
      responseText.textContent = detail.response_text || "";
    }

    function renderStats() {
      statsGrid.innerHTML = "";
      const stats = state.layerData?.matrix_stats || {};
      const items = [
        ["min", stats.min],
        ["max", stats.max],
        ["mean", stats.mean],
        ["p1", stats.p1],
        ["p95", stats.p95],
        ["p99", stats.p99],
      ];
      items.forEach(([label, value]) => {
        const box = document.createElement("div");
        box.className = "meta-item";
        box.innerHTML = "<strong>" + label + "</strong><span>" + (value == null ? "n/a" : Number(value).toExponential(4)) + "</span>";
        statsGrid.appendChild(box);
      });
    }

    function renderPromptTokens() {
      promptTokens.innerHTML = "";
      const tokens = state.layerData?.tokens || state.agentDetail?.tokens || [];
      const scores = state.layerData?.token_scores || [];
      const fragment = document.createDocumentFragment();
      tokens.forEach((token, index) => {
        const score = scores[index] ?? 0;
        const [r, g, b] = colorForValue(score, state.vmin, state.vmax);
        const span = document.createElement("span");
        span.className = "token";
        span.style.backgroundColor = "rgba(" + r + "," + g + "," + b + ",0.92)";
        span.textContent = token;
        fragment.appendChild(span);
      });
      promptTokens.appendChild(fragment);
    }

    function renderHeatmap() {
      const matrix = state.layerData?.matrix || [];
      if (!matrix.length || !matrix[0]?.length) {
        const ctx = heatmapCanvas.getContext("2d");
        heatmapCanvas.width = 1;
        heatmapCanvas.height = 1;
        ctx.clearRect(0, 0, 1, 1);
        return;
      }
      const rows = matrix.length;
      const cols = matrix[0].length;
      const offscreen = document.createElement("canvas");
      offscreen.width = cols;
      offscreen.height = rows;
      const offCtx = offscreen.getContext("2d");
      const image = offCtx.createImageData(cols, rows);
      let ptr = 0;
      for (let y = 0; y < rows; y += 1) {
        for (let x = 0; x < cols; x += 1) {
          const [r, g, b] = colorForValue(matrix[y][x], state.vmin, state.vmax);
          image.data[ptr++] = r;
          image.data[ptr++] = g;
          image.data[ptr++] = b;
          image.data[ptr++] = 255;
        }
      }
      offCtx.putImageData(image, 0, 0);
      const scale = Math.max(1, Math.floor(900 / Math.max(rows, cols)));
      heatmapCanvas.width = cols * scale;
      heatmapCanvas.height = rows * scale;
      const ctx = heatmapCanvas.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
      ctx.drawImage(offscreen, 0, 0, heatmapCanvas.width, heatmapCanvas.height);
    }

    function renderLayerVisuals() {
      renderStats();
      renderHeatmap();
      renderPromptTokens();
    }

    function syncLayerControls() {
      const layers = state.agentDetail?.available_layers || [];
      const maxIndex = Math.max(layers.length - 1, 0);
      layerSlider.min = "0";
      layerSlider.max = String(maxIndex);
      layerSlider.step = "1";
      layerSlider.value = String(state.selectedLayerIndex);
      const layerValue = layers[state.selectedLayerIndex] ?? "n/a";
      layerLabel.textContent = "Layer: " + layerValue;
      prevLayerBtn.disabled = state.selectedLayerIndex <= 0;
      nextLayerBtn.disabled = state.selectedLayerIndex >= maxIndex;
    }

    async function loadLayer() {
      const layers = state.agentDetail?.available_layers || [];
      if (!layers.length) return;
      syncLayerControls();
      const layerValue = layers[state.selectedLayerIndex];
      setStatus("Loading layer " + layerValue + "...");
      const runTag = encodeURIComponent(state.selectedRun);
      const requestUid = encodeURIComponent(state.selectedRequest);
      const agentId = encodeURIComponent(state.selectedAgent);
      const url = "/api/runs/" + runTag + "/requests/" + requestUid + "/agents/" + agentId + "/layers/" + layerValue + "?round_index=" + encodeURIComponent(state.selectedRound);
      state.layerData = await fetchJson(url);
      const stats = state.layerData.matrix_stats || {};
      state.robustMin = stats.p1 ?? stats.min ?? 0;
      state.robustMax = stats.p99 ?? stats.max ?? 1;
      applyRangeToControls(
        stats.min ?? 0,
        stats.max ?? 1,
        stats.min ?? 0,
        stats.max ?? 1,
      );
      renderLayerVisuals();
      setStatus("Showing layer " + layerValue + ".");
    }

    async function loadAgentDetail() {
      if (!state.selectedRun || !state.selectedRequest || state.selectedRound == null || !state.selectedAgent) {
        return;
      }
      const runTag = encodeURIComponent(state.selectedRun);
      const requestUid = encodeURIComponent(state.selectedRequest);
      const agentId = encodeURIComponent(state.selectedAgent);
      setStatus("Loading agent detail...");
      const url = "/api/runs/" + runTag + "/requests/" + requestUid + "/agents/" + agentId + "?round_index=" + encodeURIComponent(state.selectedRound);
      state.agentDetail = await fetchJson(url);
      state.selectedLayerIndex = 0;
      renderMetadata();
      await loadLayer();
    }

    function syncRequestControls() {
      const requests = state.runData?.requests || [];
      setSelectOptions(requestSelect, requests, "request_uid", (item) => item.request_uid);
      if (!requests.length) {
        requestSelect.innerHTML = "";
        roundSelect.innerHTML = "";
        agentSelect.innerHTML = "";
        state.selectedRequest = null;
        state.selectedRound = null;
        state.selectedAgent = null;
        return;
      }
      if (!requests.some((item) => item.request_uid === state.selectedRequest)) {
        state.selectedRequest = requests[0].request_uid;
      }
      requestSelect.value = state.selectedRequest;

      const rounds = currentRequestData()?.rounds || [];
      setSelectOptions(roundSelect, rounds, "round_index", (item) => "Round " + item.round_index);
      if (!rounds.some((item) => String(item.round_index) === String(state.selectedRound))) {
        state.selectedRound = rounds[0]?.round_index ?? null;
      }
      roundSelect.value = String(state.selectedRound);

      const agents = currentRoundData()?.agents || [];
      setSelectOptions(agentSelect, agents, "agent_id", (item) => {
        const name = item.agent_name || item.agent_id;
        return name + " [" + (item.agent_role || "role") + "]";
      });
      if (!agents.some((item) => item.agent_id === state.selectedAgent)) {
        state.selectedAgent = agents[0]?.agent_id ?? null;
      }
      agentSelect.value = state.selectedAgent || "";
    }

    async function loadRun(runTag) {
      state.selectedRun = runTag;
      setStatus("Loading run...");
      state.runData = await fetchJson("/api/runs/" + encodeURIComponent(runTag));
      syncRequestControls();
      await loadAgentDetail();
    }

    async function boot() {
      try {
        setStatus("Loading runs...");
        state.runs = await fetchJson("/api/runs");
        setSelectOptions(runSelect, state.runs, "run_tag", (item) => item.run_tag + " (" + item.call_count + " calls)");
        if (!state.runs.length) {
          setStatus("No attention visualization runs found.");
          return;
        }
        state.selectedRun = state.runs[0].run_tag;
        runSelect.value = state.selectedRun;
        await loadRun(state.selectedRun);
      } catch (error) {
        console.error(error);
        setStatus(error.message);
      }
    }

    runSelect.addEventListener("change", async (event) => {
      try {
        await loadRun(event.target.value);
      } catch (error) {
        setStatus(error.message);
      }
    });

    requestSelect.addEventListener("change", async (event) => {
      state.selectedRequest = event.target.value;
      syncRequestControls();
      try {
        await loadAgentDetail();
      } catch (error) {
        setStatus(error.message);
      }
    });

    roundSelect.addEventListener("change", async (event) => {
      state.selectedRound = event.target.value;
      syncRequestControls();
      try {
        await loadAgentDetail();
      } catch (error) {
        setStatus(error.message);
      }
    });

    agentSelect.addEventListener("change", async (event) => {
      state.selectedAgent = event.target.value;
      try {
        await loadAgentDetail();
      } catch (error) {
        setStatus(error.message);
      }
    });

    layerSlider.addEventListener("input", async (event) => {
      state.selectedLayerIndex = Number(event.target.value);
      try {
        await loadLayer();
      } catch (error) {
        setStatus(error.message);
      }
    });

    prevLayerBtn.addEventListener("click", async () => {
      if (state.selectedLayerIndex <= 0) return;
      state.selectedLayerIndex -= 1;
      try {
        await loadLayer();
      } catch (error) {
        setStatus(error.message);
      }
    });

    nextLayerBtn.addEventListener("click", async () => {
      const maxIndex = (state.agentDetail?.available_layers || []).length - 1;
      if (state.selectedLayerIndex >= maxIndex) return;
      state.selectedLayerIndex += 1;
      try {
        await loadLayer();
      } catch (error) {
        setStatus(error.message);
      }
    });

    vminInput.addEventListener("change", () => updateRangeFromInputs("vmin"));
    vmaxInput.addEventListener("change", () => updateRangeFromInputs("vmax"));
    vminSlider.addEventListener("input", () => updateRangeFromInputs("slider"));
    vmaxSlider.addEventListener("input", () => updateRangeFromInputs("slider"));

    resetRangeBtn.addEventListener("click", () => {
      applyRangeToControls(state.absMin, state.absMax, state.absMin, state.absMax);
      renderLayerVisuals();
    });

    robustRangeBtn.addEventListener("click", () => {
      applyRangeToControls(state.absMin, state.absMax, state.robustMin, state.robustMax);
      renderLayerVisuals();
    });

    boot();
  </script>
</body>
</html>
"""


def create_app(session_root: str | Path):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required to run the attention viewer. Install project dependencies first."
        ) from exc

    store = AttentionSessionStore(session_root)
    app = FastAPI(title="KVCOMM Attention Viewer")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_viewer_html())

    @app.get("/api/runs")
    async def list_runs() -> List[Dict[str, Any]]:
        return store.list_runs()

    @app.get("/api/runs/{run_tag}")
    async def get_run(run_tag: str) -> Dict[str, Any]:
        try:
            return store.get_run_index(run_tag)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_tag}/requests/{request_uid}/agents/{agent_id}")
    async def get_agent(
        run_tag: str,
        request_uid: str,
        agent_id: str,
        round_index: Optional[int] = Query(default=None),
    ) -> Dict[str, Any]:
        try:
            return store.get_agent_detail(
                run_tag=run_tag,
                request_uid=request_uid,
                agent_id=agent_id,
                round_index=round_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_tag}/requests/{request_uid}/agents/{agent_id}/layers/{layer_idx}")
    async def get_layer(
        run_tag: str,
        request_uid: str,
        agent_id: str,
        layer_idx: int,
        round_index: Optional[int] = Query(default=None),
    ) -> Dict[str, Any]:
        try:
            return store.get_layer_payload(
                run_tag=run_tag,
                request_uid=request_uid,
                agent_id=agent_id,
                round_index=round_index,
                layer_idx=layer_idx,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve KVCOMM attention visualization sessions.")
    parser.add_argument(
        "--session-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "result" / "mmlu" / "attn_heatmaps"),
        help="Root directory that contains exported attention visualization runs.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to run the attention viewer. Install it before launching this service."
        ) from exc

    app = create_app(args.session_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
