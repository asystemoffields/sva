# SVA Long-Context Extension Snapshot

This snapshot records the first long-context extension pressure tests for the frozen SmolLM2 `2x256` SVA artifact.

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Hardware: Modal H100
- 8k head-to-head app: `ap-AZDssTFUMTzUwYMabkySlR`
- Long-context recall app: `ap-W6JZ7Nc6xssoPPZ4AUJslx`
- Long-context scale-out app: `ap-Qsicm9kBNMFxtyRpefn2NR`

## 8k Head-to-Head

At SmolLM2's configured `8192` context window, the all-layer SVA socket remains very close to full attention:

| setting | loss delta | KL | top-1 agreement | logit cosine | exact/value reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shortlist=2048`, `budget=512` | `0.000907` | `0.000446` | `0.999695` | `0.991188` | `16x` fewer |

The wall-clock result is still a systems problem. In the current stock PyTorch socket, full prefill averaged `21.94 ms` and SVA prefill averaged `7331 ms`; full decode averaged `14.78 ms` and SVA decode averaged `124.21 ms`. The method-level work proxy favored SVA by `2.91x`, but the implementation path is paying too much overhead.

## Fixed-Budget Long-Context Recall

This proxy freezes the 8k artifact, extracts real SmolLM2 Q/K activations from an 8k held-out sequence, synthesizes larger key banks from that activation distribution, computes exact full-attention top keys, and measures whether SVA recovers those keys.

| context | extension | shortlist | budget | verified top-16 recall | exact/value read reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8192` | `1x` | `2048` | `512` | `0.977778` | `16x` |
| `32768` | `4x` | `2048` | `512` | `0.848611` | `64x` |
| `131072` | `16x` | `2048` | `512` | `0.657639` | `256x` |
| `1000000` | `122x` | `2048` | `512` | `0.365885` | `1953x` |

Fixed `2048/512` budgets degrade as the field grows. That is the expected pressure point: a fixed poem has to describe more terrain.

## Scale-Out Recall

The follow-up increased shortlists and budgets at `128k` and `1M`.

| context | shortlist | budget | candidate top-16 recall | verified top-16 recall | exact/value read reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `131072` | `8192` | `2048` | `0.894705` | `0.886024` | `64x` |
| `131072` | `16384` | `1024` | `0.965017` | `0.918403` | `128x` |
| `131072` | `16384` | `2048` | `0.965017` | `0.943056` | `64x` |
| `1000000` | `8192` | `2048` | `0.541927` | `0.530729` | `488x` |
| `1000000` | `16384` | `1024` | `0.677431` | `0.611545` | `977x` |
| `1000000` | `16384` | `2048` | `0.677431` | `0.645399` | `488x` |

The `128k` result is the strongest long-context signal so far: the current artifact can recover high top-key recall with a verifier budget that is still much smaller than full attention. The `1M` result shows the next invention target: the catalog needs richer or multi-scale addressing before we can expect million-token accuracy from this artifact family.

## Decode Summon Path

The inverted posting-list decode path reduced exact/value reads sharply but is slower in the current Python/PyTorch adapter. Partial H100 rows:

| variant | avg verified | exact/value read reduction | KL | top-1 agreement | decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| full attention | `8192` | `1x` | `0.000000` | `1.000000` | `19.43` |
| scan SVA `2048/512` | `512` | `16x` | `0.000017` | `1.000000` | `150.65` |
| inverted, cells `16`, min `128` | `112.69` | `72.70x` | `0.000130` | `1.000000` | `321.54` |
| inverted, cells `16`, min `256` | `208.34` | `39.32x` | `0.000094` | `1.000000` | `341.63` |
| inverted, cells `32`, min `128` | `125.64` | `65.20x` | `0.000065` | `1.000000` | `333.31` |

The quality signal is useful; the adapter path needs vectorization or a fused lookup.

## Interpretation

SVA remains strongest as a long-context extension candidate. At `8k`, full attention is too optimized for the current socket to beat on wall-clock. At `128k`, the fixed artifact recovers strong key recall with a sparse verifier budget. At `1M`, the same fixed artifact loses too much recall even when the shortlist grows to `16384`.

The next decisive test is a real long-context language benchmark that separates three questions:

1. how far full attention can run before memory or wall-clock becomes the limit,
2. whether SVA preserves task accuracy beyond that limit,
3. how large the catalog and verifier budget must become to preserve full-attention behavior.

For method work, the next target is a richer catalog: multi-scale codes, layer-aware budgets, and a shortlist objective trained directly for the target context scale.
