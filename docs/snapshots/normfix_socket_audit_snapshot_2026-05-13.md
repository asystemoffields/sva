# Attention-Input Normalization Fix Snapshot - 2026-05-13

This snapshot records the investigation into the suspicious selective-layer result where layers `11`, `16`, and `19` appeared to reject SVA while many neighboring layers tolerated it.

## Finding

The artifact builder was training Q/K routes from layer-boundary hidden states. Hugging Face's Llama decoder layer applies `input_layernorm` before self-attention:

```text
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)
```

That means SVA artifacts were trained on pre-norm hidden states but served on post-norm attention inputs. The fix is to apply `layer.input_layernorm(hidden_states[layer_idx])` before deriving Q/K training targets and hidden-state route inputs.

Patched files:

- `experiments/sva_pretrained_socket_test.py`
- `experiments/sva_learned_ranker_test.py`
- `experiments/sva_real_qk_address_sweep.py`
- `experiments/sva_million_stream_sim.py`

## Norm-Fixed Cliff Rerun

All runs used `HuggingFaceTB/SmolLM2-135M-Instruct`, `seq_len=2048`, QK routing, `4x64` coarse PQ, `coarse_shortlist=1024`, `budget=512`, `ranker_train_steps=160`, `coarse_hard_steps=80`, hard pool `512`, and attention-weighted codebooks with boost `4`.

| condition | training | socketed count | loss_delta | KL | top1 agreement | logit cosine | verified top16 recall |
|---|---|---:|---:|---:|---:|---:|---:|
| base_15 | teacher | 15 | 0.000000 | 0.000354 | 0.993161 | 0.998981 | 0.999852 |
| add_14 | teacher | 16 | 0.000000 | 0.000361 | 0.993649 | 0.998957 | 0.999838 |
| add_16 | teacher | 16 | 0.000000 | 0.000347 | 0.994138 | 0.998949 | 0.999859 |
| add_19 | teacher | 16 | 0.000000 | 0.000340 | 0.995115 | 0.999007 | 0.999847 |
| add_20 | teacher | 16 | 0.000000 | 0.000358 | 0.995115 | 0.998786 | 0.999813 |
| add_23 | teacher | 16 | 0.000000 | 0.000354 | 0.993161 | 0.998994 | 0.999845 |
| add_14_16 | teacher | 17 | 0.000000 | 0.000344 | 0.994626 | 0.998913 | 0.999845 |
| add_19_20 | teacher | 17 | 0.000000 | 0.000341 | 0.995115 | 0.998450 | 0.999801 |
| frontier_20 | teacher | 20 | 0.000000 | 0.000355 | 0.995115 | 0.998173 | 0.999799 |
| add_14_16 | progressive | 17 | 0.000000 | 0.000363 | 0.995603 | 0.998778 | 0.999864 |
| add_19_20 | progressive | 17 | 0.000000 | 0.000344 | 0.995603 | 0.998236 | 0.999834 |
| frontier_20 | progressive | 20 | 0.000000 | 0.000353 | 0.992672 | 0.997979 | 0.999817 |

## Norm-Fixed All-Layer Rerun

| condition | training | socketed count | loss_delta | KL | top1 agreement | logit cosine | verified top16 recall |
|---|---|---:|---:|---:|---:|---:|---:|
| all layers | teacher | 30 | 0.000000 | 0.000362 | 0.994626 | 0.997908 | 0.999689 |
| all layers | progressive | 30 | 0.000000 | 0.000361 | 0.996092 | 0.997757 | 0.999703 |

## Readout

The layer-specific collapse was a harness/interface bug. The suspicious layers stopped being suspicious once artifacts were trained on the same normalized attention inputs that the socket uses at inference time.

This invalidates the earlier interpretation that layers `11`, `16`, or `19` were intrinsically fragile under SVA. The older selective-layer, cliff, fallback, and admission snapshots remain useful as a debugging trail, but their mechanism interpretation is stale.

The corrected result is much stronger than the prior one: in this `seq_len=2048` SmolLM2 socket harness, SVA can replace all 30 attention layers while preserving the model's next-token distribution extremely closely.

## Next Step

The next pressure tests should move away from layer fallback and toward scale:

1. Repeat the corrected all-layer socket at longer contexts, especially SmolLM2's configured `8192` window.
2. Reduce verifier budget and shortlist to find the real quality/cost frontier after the normalization fix.
3. Re-run the million-token lookup simulations with normalized Q/K samples.
4. Keep a regression check that verifies artifact Q/K extraction matches the live attention input path.
