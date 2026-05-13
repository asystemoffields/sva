# SVA Causal Sequence Snapshot

Date: 2026-05-13

## Test

This checkpoint moves SVA from static retrieval into a causal cache. Each timestep writes one key/value page into incremental SVA tables. Later queries must recover one prior page using only pages already written.

The benchmark uses synthetic clustered keys, noisy queries, and exact causal attention as the teacher.

## Shared Setup

- 8 sequences
- 1024 tokens per sequence
- average prefix length: 512 pages
- query noise: 0.10
- verifier budget: 16
- probe radius: 1
- prefilter: 32 dimensions, budget 128

## 10-Bit Addresses

```text
method                                      top1_target  cos_teacher  avg_summoned  avg_exact_scored  avg_verified
full_causal_attention                       0.9988       1.0000       512.0         512.0             512.0
sva_causal_probe1_8x10                      0.9556       0.9588       54.4          54.4              14.8
sva_causal_probe1_prefilter32d128_8x10      0.9555       0.9578       55.4          55.2              14.8
sva_causal_probe1_16x10                     0.9965       0.9970       97.7          97.7              15.4
sva_causal_probe1_prefilter32d128_16x10     0.9961       0.9970       98.4          85.7              15.4
sva_causal_probe1_24x10                     0.9984       0.9994       135.6         135.6             15.5
sva_causal_probe1_prefilter32d128_24x10     0.9985       0.9995       135.7         97.5              15.6
```

## 12-Bit Addresses

```text
method                                      top1_target  cos_teacher  avg_summoned  avg_exact_scored  avg_verified
full_causal_attention                       0.9988       1.0000       512.0         512.0             512.0
sva_causal_probe1_8x12                      0.8755       0.8824       20.6          20.6              12.7
sva_causal_probe1_prefilter32d128_8x12      0.8799       0.8868       21.5          21.5              12.8
sva_causal_probe1_16x12                     0.9796       0.9809       39.2          39.2              14.4
sva_causal_probe1_prefilter32d128_16x12     0.9806       0.9819       38.2          38.2              14.4
sva_causal_probe1_24x12                     0.9954       0.9962       53.8          53.8              14.8
sva_causal_probe1_prefilter32d128_24x12     0.9957       0.9962       54.2          54.1              14.9
```

## Interpretation

SVA transfers cleanly into the causal setting. The 12-bit, 24-table probe variant is especially attractive: it stays close to full causal attention while summoning only about 54 pages from an average prefix of 512.

The prefilter matters most when the raw summoned set grows beyond its budget. In this causal run, 12-bit addresses already keep the summoned set below 128, so the prefilter preserves behavior rather than changing cost.

## Next Risk

The next risk is learned keys and queries. These synthetic tests assume the query is a noisy version of the target key. A stronger test should put SVA inside a tiny trainable causal model and ask whether learned representations naturally make usable summon addresses.
