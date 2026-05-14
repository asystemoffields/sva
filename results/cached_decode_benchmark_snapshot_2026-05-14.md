# Cached Decode Benchmark Snapshot - 2026-05-14

This snapshot records the first cached-key deployment decode benchmark for SVA.

## Run

- Modal app: `ap-9A7yUIcAPYWfhAsJhtAFo8`
- Function call: `fc-01KRJC642QJ5ZAGGQKYZTDE5ZB`
- Dashboard: https://modal.com/id/fc-01KRJC642QJ5ZAGGQKYZTDE5ZB
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Hardware: H100
- Context: `8192`
- Eval: 4 held-out built-in documents
- Layers: all 30
- Route source: Q/K
- Artifact training: teacher
- Quality samples: 128 query positions per held-out document
- Timing query counts: `1`, `4`, `16`

This benchmark precomputes the key-side low-rank catalog and product codes once per layer, then measures decode-shaped lookup. That separates the deployment lookup from the full socket harness, which rebuilds product codes inside every forward pass.

## Quality Results

| shortlist | budget | top items | candidate top16 recall | verified top16 recall |
|---:|---:|---:|---:|---:|
| 512 | 128 | 2,211,840 | 0.957842 | 0.954132 |
| 512 | 256 | 2,211,840 | 0.957842 | 0.957596 |
| 512 | 512 | 2,211,840 | 0.957842 | 0.957842 |
| 1024 | 128 | 2,211,840 | 0.988075 | 0.981061 |
| 1024 | 256 | 2,211,840 | 0.988075 | 0.987072 |
| 1024 | 512 | 2,211,840 | 0.988075 | 0.988018 |
| 2048 | 128 | 2,211,840 | 0.998307 | 0.989134 |
| 2048 | 256 | 2,211,840 | 0.998307 | 0.996600 |
| 2048 | 512 | 2,211,840 | 0.998307 | 0.998094 |

## Timing Readout

The average cached catalog build time was about `7.99 ms` per layer/document at `8192` tokens.

The current PyTorch cached decode path is still slower than full attention at `8192`:

| shortlist | budget | query count | full decode ms avg | SVA decode ms avg | full/SVA |
|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 1 | 0.308 | 3.110 | 0.099 |
| 1024 | 512 | 1 | 0.300 | 3.109 | 0.096 |
| 2048 | 512 | 1 | 0.303 | 3.128 | 0.097 |
| 512 | 128 | 16 | 0.528 | 3.228 | 0.164 |
| 1024 | 512 | 16 | 0.529 | 3.257 | 0.162 |
| 2048 | 512 | 16 | 0.525 | 3.368 | 0.156 |

## Readout

The quality signal survives the cached decode setup: `2048/512` reaches verified top-16 recall `0.998094`, and `1024/512` reaches `0.988018`.

The systems signal is sharper now too. At an `8192` context, full attention is still cheap enough and optimized enough that the current Python/PyTorch SVA lookup loses on latency. The SVA systems case depends on longer contexts and a real serving implementation: cached key-side codes, fused table lookups, and avoiding per-query Python loops.

## Next Step

The next definitive systems test should scale the cached decode lookup to synthetic million-token key caches using the same measured shape, while keeping the quality frontier anchored by the 8192 held-out deployment result.
