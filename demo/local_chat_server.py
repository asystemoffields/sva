"""Local SVA chat demo server.

Run from the repository root:

    python demo/local_chat_server.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import SVALlamaPatcher, patch_llama_attention


DEFAULT_ARTIFACT_DIR = Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1")
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SVA Local Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #191d24;
      --panel-2: #20252e;
      --line: #353c49;
      --text: #f4f1e8;
      --muted: #b7bdc8;
      --accent: #62d2a2;
      --accent-2: #e2b86b;
      --danger: #ef7d7d;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #151922;
      display: grid;
      grid-template-rows: auto 180px 1fr;
      min-width: 0;
    }
    .brand {
      padding: 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    canvas {
      width: 100%;
      height: 180px;
      display: block;
      border-bottom: 1px solid var(--line);
      background: #10141b;
    }
    .stats {
      padding: 14px 18px;
      display: grid;
      align-content: start;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .stat-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(255,255,255,.06);
      padding-bottom: 8px;
    }
    .stat-row strong {
      color: var(--text);
      font-weight: 600;
      overflow-wrap: anywhere;
      text-align: right;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: #111318;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: #171b22;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    button {
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      height: 38px;
      min-width: 38px;
      border-radius: 7px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 0 12px;
      cursor: pointer;
      font: inherit;
      font-size: 14px;
    }
    button:hover { border-color: var(--accent); }
    button:disabled {
      cursor: wait;
      opacity: .58;
    }
    .primary {
      border-color: rgba(98, 210, 162, .55);
      background: #214036;
    }
    svg { width: 17px; height: 17px; flex: 0 0 auto; }
    #messages {
      min-height: 0;
      overflow: auto;
      padding: 20px clamp(16px, 4vw, 46px);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 880px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      line-height: 1.48;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .user {
      align-self: flex-end;
      background: #22302c;
      border-color: rgba(98, 210, 162, .32);
    }
    .assistant {
      align-self: flex-start;
      background: var(--panel);
    }
    .error {
      align-self: center;
      background: #3a2024;
      border-color: rgba(239, 125, 125, .5);
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 14px 18px 18px;
      border-top: 1px solid var(--line);
      background: #171b22;
    }
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #11151c;
      color: var(--text);
      padding: 12px 13px;
      line-height: 1.38;
      font: inherit;
      font-size: 15px;
    }
    textarea:focus {
      outline: 2px solid rgba(98, 210, 162, .32);
      border-color: var(--accent);
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(82px, 1fr));
      gap: 8px;
      padding: 0 18px 14px;
      background: #171b22;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #11151c;
      color: var(--text);
      padding: 8px 9px;
      font: inherit;
    }
    @media (max-width: 780px) {
      .app { grid-template-columns: 1fr; }
      aside { display: none; }
      header { align-items: flex-start; flex-direction: column; }
      form { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <h1>SVA Local Chat</h1>
        <span class="pill" id="loaded">cold</span>
      </div>
      <canvas id="field" width="640" height="360"></canvas>
      <div class="stats">
        <div class="stat-row"><span>Model</span><strong id="model">SmolLM2</strong></div>
        <div class="stat-row"><span>Device</span><strong id="device">-</strong></div>
        <div class="stat-row"><span>Queries</span><strong id="queries">0</strong></div>
        <div class="stat-row"><span>Summoned</span><strong id="summoned">-</strong></div>
        <div class="stat-row"><span>Verified</span><strong id="verified">-</strong></div>
      </div>
    </aside>
    <main>
      <header>
        <div class="status" id="status">Ready</div>
        <div class="toolbar">
          <button id="clear" type="button" title="Clear">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/></svg>
          </button>
        </div>
      </header>
      <section id="messages" aria-live="polite"></section>
      <div class="controls">
        <label>Tokens<input id="tokens" type="number" min="8" max="512" step="8" value="96"></label>
        <label>Temperature<input id="temperature" type="number" min="0" max="2" step="0.05" value="0.7"></label>
        <label>Top-p<input id="top_p" type="number" min="0.05" max="1" step="0.05" value="0.9"></label>
      </div>
      <form id="chat">
        <textarea id="prompt" rows="2" placeholder="Message SVA"></textarea>
        <button class="primary" id="send" type="submit" title="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4Z"/></svg>
          <span>Send</span>
        </button>
      </form>
    </main>
  </div>
  <script>
    const messages = [];
    const el = (id) => document.getElementById(id);
    const messagesEl = el('messages');
    const statusEl = el('status');
    const sendEl = el('send');
    const promptEl = el('prompt');

    function addMessage(role, text) {
      messages.push({ role, content: text });
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      node.textContent = text;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function showError(text) {
      const node = document.createElement('div');
      node.className = 'msg error';
      node.textContent = text;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setStats(data) {
      if (data.model_id) el('model').textContent = data.model_id.split('/').pop();
      if (data.device) el('device').textContent = data.device;
      if (typeof data.loaded === 'boolean') el('loaded').textContent = data.loaded ? 'loaded' : 'cold';
      const stats = data.stats || {};
      if (stats.queries !== undefined) el('queries').textContent = Math.round(stats.queries).toString();
      if (Number.isFinite(stats.avg_summoned)) el('summoned').textContent = stats.avg_summoned.toFixed(1);
      if (Number.isFinite(stats.avg_verified)) el('verified').textContent = stats.avg_verified.toFixed(1);
    }

    async function refreshStatus() {
      const response = await fetch('/api/status');
      setStats(await response.json());
    }

    el('clear').addEventListener('click', () => {
      messages.length = 0;
      messagesEl.replaceChildren();
      statusEl.textContent = 'Ready';
      promptEl.focus();
    });

    el('chat').addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = '';
      addMessage('user', text);
      sendEl.disabled = true;
      statusEl.textContent = 'Thinking';
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages,
            max_new_tokens: Number(el('tokens').value),
            temperature: Number(el('temperature').value),
            top_p: Number(el('top_p').value)
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        addMessage('assistant', data.reply);
        setStats(data);
        statusEl.textContent = 'Ready';
      } catch (error) {
        showError(error.message || String(error));
        statusEl.textContent = 'Error';
      } finally {
        sendEl.disabled = false;
        promptEl.focus();
      }
    });

    promptEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        el('chat').requestSubmit();
      }
    });

    const canvas = el('field');
    const ctx = canvas.getContext('2d');
    const points = Array.from({ length: 80 }, (_, i) => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: 1 + Math.random() * 2,
      phase: i * 0.37
    }));
    function draw(t) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#10141b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const p of points) {
        const pulse = 0.45 + 0.55 * Math.sin(t * 0.0018 + p.phase);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r + pulse * 2.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(98, 210, 162, ${0.16 + pulse * 0.42})`;
        ctx.fill();
      }
      ctx.strokeStyle = 'rgba(226, 184, 107, .32)';
      ctx.lineWidth = 1;
      for (let i = 0; i < points.length; i += 7) {
        const a = points[i];
        const b = points[(i * 13 + 5) % points.length];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
    refreshStatus().catch(() => {});
  </script>
</body>
</html>
"""


class DemoState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.loaded = False
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.tokenizer: Any = None
        self.model: Any = None
        self.patcher: SVALlamaPatcher | None = None

    def load(self) -> None:
        if self.loaded:
            return
        if self.args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)
        dtype_map = {
            "auto": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.patcher = patch_llama_attention(
            self.model,
            self.args.artifact_dir,
            shortlist=self.args.shortlist,
            budget=self.args.budget,
        )
        self.loaded = True

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model_id": self.args.model_id,
            "device": str(self.device),
            "stats": self.patcher.stats.summary() if self.patcher is not None else {},
        }

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            assert self.model is not None
            assert self.tokenizer is not None
            assert self.patcher is not None
            self.patcher.reset_stats()
            self.patcher.reset_catalogs()
            raw_messages = payload.get("messages", [])
            messages = [
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in raw_messages
                if str(item.get("content", "")).strip()
            ][-16:]
            if not messages:
                raise ValueError("No messages supplied.")

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            max_new_tokens = max(1, min(int(payload.get("max_new_tokens", self.args.max_new_tokens)), 512))
            temperature = max(0.0, float(payload.get("temperature", self.args.temperature)))
            top_p = min(1.0, max(0.05, float(payload.get("top_p", self.args.top_p))))
            max_input_tokens = max(64, int(self.args.context_length) - max_new_tokens)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(self.device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
            else:
                generation_kwargs.update({"do_sample": False})

            with torch.no_grad():
                output = self.model.generate(**inputs, **generation_kwargs)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return {
                "reply": reply,
                "model_id": self.args.model_id,
                "device": str(self.device),
                "stats": self.patcher.stats.summary(),
                "loaded": self.loaded,
            }


def make_handler(state: DemoState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SVALocalChat/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                self.send_html(HTML)
                return
            if self.path == "/api/status":
                self.send_json(state.status())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body) if body else {}
                self.send_json(state.chat(payload))
            except Exception as exc:
                self.send_json(
                    {"error": str(exc), "traceback": traceback.format_exc(limit=4)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local browser chat UI backed by SmolLM2 + SVA.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--shortlist", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = DemoState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"sva_chat_url,http://{args.host}:{args.port}", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"artifact_dir,{args.artifact_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
