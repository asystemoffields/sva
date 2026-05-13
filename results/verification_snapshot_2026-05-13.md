# SVA Verification Snapshot

Date: 2026-05-13

## Mechanism Name

**Summon-Verify Attention (SVA)**

Pages self-index into several content-addressed lookup tables. Queries activate the same tables, summon a small candidate set, and then run exact attention over only those candidates.

## Strongest Result So Far

8192 stored pages, 1024 queries, 16-candidate verifier budget, 10-bit tables, query noise 0.05, attention logit scale 16.

Random clustered keys:

```text
method             top1    cos_teacher  avg_candidates
full_attention     1.0000  1.0000       8192.0
coarse_bank        0.5317  0.5469       15.8
sva_16x10          0.9932  0.9716       16.0
sva_24x10          0.9995  0.9785       16.0
```

Entity-attribute binding keys:

```text
method             top1    cos_teacher  avg_candidates
full_attention     1.0000  1.0000       8192.0
coarse_bank        0.5552  0.5820       15.7
sva_16x10          0.9907  0.9721       16.0
sva_24x10          0.9995  0.9859       16.0
```

## Budget Stress

4096 stored pages, 1024 queries, 8-candidate verifier budget, query noise 0.05, logit scale 16.

```text
method             top1    cos_teacher  avg_candidates
full_attention     1.0000  1.0000       4096.0
coarse_bank        0.5335  0.5454       7.9
sva_8x10           0.9248  0.9176       8.0
sva_16x10          0.9941  0.9825       8.0
sva_24x10          1.0000  0.9884       8.0
```

## Noise Stress

4096 stored pages, 1024 queries, 16-candidate verifier budget, query noise 0.10, logit scale 16.

Random clustered keys:

```text
method             top1    cos_teacher  avg_candidates
full_attention     0.9937  1.0000       4096.0
coarse_bank        0.2988  0.3317       15.1
sva_16x10          0.7808  0.7989       16.0
sva_24x10          0.8926  0.8888       16.0
sva_32x10          0.9385  0.9290       16.0
```

Entity-attribute binding keys:

```text
method             top1    cos_teacher  avg_candidates
full_attention     0.9287  1.0000       4096.0
coarse_bank        0.2588  0.3279       14.7
sva_16x10          0.7124  0.8018       16.0
sva_24x10          0.8213  0.8879       16.0
sva_32x10          0.8809  0.9426       16.0
```

## Softer Teacher Stress

4096 binding pages, 1024 queries, 16-candidate verifier budget, query noise 0.05, logit scale 8.

```text
method             top1    cos_teacher  avg_candidates
full_attention     0.9995  1.0000       4096.0
coarse_bank        0.4990  0.5096       15.2
sva_8x10           0.9092  0.8176       16.0
sva_16x10          0.9927  0.9149       16.0
sva_24x10          0.9990  0.9360       16.0
```

## Interpretation

SVA is a strong go on synthetic nearest-neighbor attention where the teacher is peaky. It still works under a tighter verifier budget and scales cleanly from 4096 to 8192 stored pages in this benchmark.

The main pressure point is noisy or ambiguous lookup. More tables recover much of the loss, but the next design question is how to learn better projections so the candidate set becomes semantic instead of merely geometric.
