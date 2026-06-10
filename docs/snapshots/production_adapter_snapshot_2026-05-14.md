# Production Adapter Snapshot - 2026-05-14

This snapshot records the first production-facing SVA adapter slice.

## Implemented

- Added a real `sva` Python package.
- Added artifact loading and validation independent of the experiment scripts.
- Added reversible Llama-family attention patching.
- Added cached decode catalog reuse: prefill builds the SVA key catalog, decode appends new key codes.
- Added a local browser chat demo backed by SmolLM2 plus the exported SVA artifact.
- Added no-download adapter smoke tests using a tiny random Llama model.

## Local Demo

Server:

```powershell
python demo\local_chat_server.py
```

URL:

```text
http://127.0.0.1:8765
```

The running server verified `/api/chat` with the local artifact:

```text
reply: Hello!
device: cpu
queries: 10530
avg_summoned: 20.0
avg_verified: 20.0
```

## 8k Production Target

At 8k context, SVA has to win by changing the work being done:

- Verify fewer keys than dense attention scores.
- Read fewer values than dense attention mixes.
- Reuse key-side catalogs across cached decode steps.
- Spend large budgets only on heads, layers, and queries that need them.
- Preserve full-attention behavior under an explicit loss/KL agreement target.

The current adapter removes repeated decode catalog construction. The next method-level target is adaptive verification: train and evaluate per-layer/head/query budgets so easy queries spend `64-128` exact scores while hard queries spend `512-2048`.

## Long-Context Implication

For million-token contexts, dense attention pays against every cached token. SVA pays for a compact catalog pass plus exact verification over a bounded candidate set. The current no-custom-kernel million-token synthetic result already showed a one-query opening; the path to production is to make the 8k path stop overpaying while preserving the long-context scaling advantage.
