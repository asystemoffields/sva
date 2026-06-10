# Summon-Verify Attention (SVA)

SVA is a sparse replacement for transformer attention built around content-addressed
lookup: every key/value page **writes** itself into cheap lookup tables, a query
activates the same addresses and **summons** a small candidate set, and exact
dot-product attention **verifies** only those candidates. The write address and the
read address are the same object — the pages summon themselves, so there is no
separately learned retriever to keep consistent with the cache. In its current form,
SVA reversibly patches Llama-family Hugging Face attention layers (keeping the
pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits) with frozen per-layer
artifacts: a rank-64 learned Q/K scorer served through coarse product-quantized codes,
followed by an exact attention verifier over the shortlist. Everything below was
measured on `HuggingFaceTB/SmolLM2-135M-Instruct` and synthetic million-token caches,
on H100 with stock PyTorch (no custom kernels).

## Headline results

| Result | Setup | Numbers |
| --- | --- | --- |
| All 30 attention layers replaced, calibration context | SmolLM2-135M, 2048 tokens, `4x64` coarse PQ, shortlist 1024, budget 512 | loss delta `0.000000`, KL `0.000362`, top-1 agreement `99.46%`, verified top-16 recall `0.9997` |
| All 30 layers, frozen artifacts, held-out documents, full configured window | 8192 tokens, `2x256` artifact, shortlist 2048, budget 512 | KL `0.000481`, top-1 `99.98%`, top-16 recall `0.9987`, 16x fewer exact scores and value reads |
| Late-layer socket at 4x the training window | 32768-token passkeys, SVA in layers 26-29 only, scan summon 8192/2048, 9 key/placement cases | answer KL `0.005547`, top-1 `96.8%`, answer NLL delta `0.011` |
| Tight budget + 110k-param distilled adapter | 32768 tokens, late4 socket, 512/128, 24 held-out passkey cases | answer KL `0.0319`, top-1 `94.6%`, **256x** fewer decode exact reads in SVA layers (with indexed static summon instead of scan: KL `0.0386`, top-1 `91.7%`, decode slowdown 1.29x) |
| Million-token decode lookup, no custom kernels | synthetic 1M-key cache, H100, q=1, 2048/512 | `0.65 ms`/query (`1x256` codes) to `1.03 ms` (`4x64`) vs `2.08 ms` full attention |

Sources: [`normfix_socket_audit`](docs/snapshots/normfix_socket_audit_snapshot_2026-05-13.md),
[`full_deployment_8192`](docs/snapshots/full_deployment_8192_snapshot_2026-05-14.md),
[`passkey_late4_robustness`](docs/snapshots/passkey_late4_robustness_snapshot_2026-05-14.md),
[`late4_answerce_broad_panel`](docs/snapshots/late4_answerce_broad_panel_snapshot_2026-05-14.md),
[`late4_answerce_static_tail_profile`](docs/snapshots/late4_answerce_static_tail_profile_snapshot_2026-05-14.md),
[`compact_summon_frontier`](docs/snapshots/compact_summon_frontier_snapshot_2026-05-14.md).
Every number in this README has a dated snapshot in [`docs/snapshots/`](docs/snapshots/)
with the full run configuration.

## How it works

1. **Write.** Each key projects into a learned rank-64 space (per layer, per head) and
   is assigned coarse product-quantization codes (e.g. `2` subspaces x `256` codewords).
   The codes are the key's addresses. Artifacts — low-rank Q/K projections, per-head
   logit scales, coarse codebooks — are trained once against full-attention top-k
   labels on a calibration stream, then frozen.
2. **Summon.** A query projects through the same learned map and scores the coarse
   codes to shortlist candidates (e.g. 2048 of 8192+ cached keys), either by vectorized
   scan or through a static inverted index over the code cells. The query reads the
   addresses the keys wrote — no separate retriever.
3. **Verify.** Exact attention (real Q·K, softmax, value aggregation) runs over only
   the top `budget` summoned candidates (e.g. 512, or 128 in the tight-budget socket).
   The verifier is exact, so quality failures are always summon-side and measurable as
   top-k recall.

**The deployment finding:** replacing all layers is essentially lossless at the 8k
training window but drifts at 32k, and the drift enters through early-layer prefill —
replacing layers 0-25 alone reproduces nearly all of the all-layer damage (KL `1.53`
vs `1.56`), while replacing only layers 26-29 stays at KL `0.0055`. The production
shape is therefore: full attention builds the early/middle hidden-state stream, SVA
replaces the late layers where decode reads are concentrated, and a small residual
adapter (distilled on answer-token logits) recovers tight budgets. A harness lesson
worth keeping: an apparent "fragile layer" result was actually an interface bug
(artifacts trained on pre-norm hidden states, served on `input_layernorm` outputs) —
see the [normfix snapshot](docs/snapshots/normfix_socket_audit_snapshot_2026-05-13.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Patch a model with the bundled artifact (the `sva/` package is the production path):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sva import patch_llama_attention

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager").eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

patcher = patch_llama_attention(
    model,
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    shortlist=2048,
    budget=512,
)
# ... model.generate(...) as usual; cached decode reuses the SVA key catalog ...
print(patcher.stats.summary())
patcher.unpatch()  # or use `with patch_llama_attention(...) as patcher:`
```

Run the browser chat demo (serves immediately; downloads SmolLM2-135M on first message):

```bash
python demo/local_chat_server.py
# open http://127.0.0.1:8765
```

Run the no-download test suite and the standalone toy benchmark:

```bash
python -m unittest discover -s tests -v
python experiments/sva_kill_test.py --task binding --trials 2 --tables 8 16 24 \
    --bits 10 --budget 16 --query-noise 0.05 --logit-scale 16
```

The H100 sweeps behind the snapshots run on Modal via the `modal_h100_*.py` runners;
see [`docs/h100_runbook.md`](docs/h100_runbook.md).

## Repository map

| Path | Contents |
| --- | --- |
| `sva/` | Production package: artifact loading/validation (`artifacts.py`), reversible Llama attention patching with scan and static-inverted cached decode (`llama.py`), lookup ops (`ops.py`), runtime read accounting (`stats.py`) |
| `tests/` | No-download unit tests for the adapter (tiny random Llama + tiny bundle) |
| `demo/` | Local browser chat server running SmolLM2 with the SVA artifact |
| `experiments/` | 41 standalone benchmark harnesses — see [`experiments/README.md`](experiments/README.md) for script → question → key result |
| `modal_h100_*.py` (root) | Modal H100 runners, one per sweep, each wrapping an `experiments/` harness |
| `results/hf_artifacts/` | Frozen artifact bundles (3 full `2x256` profiles for SmolLM2-135M) and 3 late4 512/128 residual adapters |
| `docs/snapshots/` | 80 dated result snapshots with full configs — [indexed by phase](docs/snapshots/README.md) |
| `docs/research_log.md` | The chronological narrative, including negative results |
| `docs/h100_runbook.md` | Modal launch commands |
| `notes/`, `side_tracks/` | Pre-SVA findings, million-token scaling targets, hierarchical-tree and tree-sitter side tracks |
| `scripts/` | PowerShell Modal launchers (this work ran from a Windows host) |

## Limitations

- **One model, one scale.** Everything language-facing is SmolLM2-135M-Instruct.
  Nothing here is verified on larger models, other families, or GQA configurations
  beyond SmolLM2's.
- **Quality at 32k+ is layer-selective.** All-layer replacement is only verified
  lossless to 8192 tokens (the model's configured window). At 32k the verified result
  is the late4 socket; early-layer replacement at long context is an open problem that
  would need long-context training.
- **Million-token numbers are synthetic.** The 0.65-1.0 ms lookups and the 1M-key
  recall figures use synthetic caches; no end-to-end million-token language run exists.
  At 1M keys the frozen 8k artifact's recall drops to `0.645` even at 16384/2048, so
  catalog capacity is the open method problem.
- **Wall-clock wins are decode-side and setting-specific.** With stock PyTorch, decode
  is faster than full attention at million-token scale (q=1-4) and in the 32k late4
  scan harness, but the 8k all-layer adapter is slower than optimized full attention,
  batched queries (q=16) lose, and **prefill is 25x slower** in the static-inverted
  socket because summon still traverses the context. No fused/custom kernel exists yet.
- **Passkey-shaped evaluation.** Long-context language results are passkey retrieval
  panels (up to 24 held-out cases), not broad benchmarks. Held-out-document perplexity
  results exist only to 8192.
- **Snapshots are point-in-time.** Earlier snapshots' interpretations are sometimes
  revised by later ones (kept deliberately; the index flags the stale ones).

## Citing and contact

If you use SVA or build on these results:

```bibtex
@misc{hill2026sva,
  author = {Hill, Alex},
  title  = {Summon-Verify Attention: content-addressed sparse attention where pages summon themselves},
  year   = {2026},
  url    = {https://github.com/asystemoffields/sva}
}
```

Issues and discussions on this repository are the best contact channel
(GitHub: [asystemoffields](https://github.com/asystemoffields)).
