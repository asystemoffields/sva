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

## Learned Table Selection

I added a split-train selection variant. It samples a pool of random SVA tables, greedily chooses the tables that recover held-out training queries best, then evaluates on fresh queries.

With query noise 0.10, 4096 stored pages, 16-candidate verifier budget, and 256 candidate tables in the selection pool:

Random clustered keys:

```text
method             top1    cos_teacher  avg_summoned  avg_candidates
sva_16x10          0.7783  0.7943       103.6         16.0
sva_selected_16x10 0.7910  0.8067       109.9         16.0
sva_32x10          0.9424  0.9284       192.3         16.0
sva_selected_32x10 0.9512  0.9359       199.0         16.0
```

Entity-attribute binding keys:

```text
method             top1    cos_teacher  avg_summoned  avg_candidates
sva_16x10          0.7207  0.8055       140.5         16.0
sva_selected_16x10 0.7461  0.8191       139.7         16.0
sva_32x10          0.8711  0.9366       261.4         16.0
sva_selected_32x10 0.8901  0.9492       274.5         16.0
```

Interpretation: learned table selection is a real but modest improvement. It nudges recall and teacher cosine upward, especially under noisy lookup. It does not change the mechanism's cost profile enough by itself.

## Adjacent-Bucket Probing

I added a query-side probing variant. Instead of summoning only the exact LSH bucket for each table, the query also summons buckets one Hamming step away. The verifier still uses only the top 16 exact dot-product candidates.

Query noise 0.10, 4096 stored pages, 16-candidate verifier budget, logit scale 16, 10-bit tables:

Random clustered keys:

```text
method          top1    cos_teacher  avg_summoned  avg_candidates
full_attention  0.9937  1.0000       4096.0        4096.0
sva_8x10        0.5376  0.5838       53.8          16.0
sva_probe1_8x10 0.9482  0.9377       431.9         16.0
sva_16x10       0.7808  0.7989       102.1         16.0
sva_probe1_16x10 0.9902 0.9741       773.4         16.0
sva_24x10       0.8926  0.8888       150.8         16.0
sva_probe1_24x10 0.9937 0.9775       1073.8        16.0
```

Entity-attribute binding keys:

```text
method          top1    cos_teacher  avg_summoned  avg_candidates
full_attention  0.9287  1.0000       4096.0        4096.0
sva_8x10        0.5068  0.5940       69.8          16.0
sva_probe1_8x10 0.8960  0.9587       533.9         16.0
sva_16x10       0.7124  0.8018       143.0         16.0
sva_probe1_16x10 0.9263 0.9924       975.9         16.0
sva_24x10       0.8213  0.8879       193.9         16.0
sva_probe1_24x10 0.9282 0.9946       1264.6        16.0
```

With 12-bit tables, probing trades some recall for lower raw summoning cost:

```text
method           top1    cos_teacher  avg_summoned  avg_candidates
sva_probe1_8x12  0.8101  0.8763       203.7         16.0
sva_probe1_16x12 0.9141  0.9745       406.6         16.0
```

Interpretation: adjacent-bucket probing is the strongest robustness result so far. The tradeoff is that exact scoring currently happens over hundreds to about a thousand summoned candidates before the 16-item verifier budget. That makes the next cost question very specific: can SVA add a cheap pre-verifier or adaptive probing rule that preserves probe robustness while reducing raw summoned count?

## Cheap Prefilter

I added a projected-space prefilter. After SVA summons candidates, the query scores them in a small random feature space and keeps only a fixed number for exact full-dimensional scoring. The final verifier still emits a 16-candidate exact attention result.

Query noise 0.10, 4096 stored pages, 16-candidate verifier budget, logit scale 16, 10-bit tables, probe radius 1, 32-dimensional prefilter:

Random clustered keys:

```text
method                         top1    cos_teacher  avg_summoned  avg_exact_scored  avg_candidates
sva_probe1_8x10                0.9468  0.9391       427.6         427.6             16.0
sva_probe1_prefilter32d128_8x10 0.9468 0.9389       427.6         128.0             16.0
sva_probe1_16x10               0.9888  0.9736       779.2         779.2             16.0
sva_probe1_prefilter32d128_16x10 0.9888 0.9725      779.2         128.0             16.0
```

Entity-attribute binding keys:

```text
method                         top1    cos_teacher  avg_summoned  avg_exact_scored  avg_candidates
sva_probe1_8x10                0.8877  0.9537       521.3         521.3             16.0
sva_probe1_prefilter32d128_8x10 0.8867 0.9509       521.3         128.0             16.0
sva_probe1_16x10               0.9272  0.9920       968.3         968.3             16.0
sva_probe1_prefilter32d128_16x10 0.9233 0.9836      968.3         128.0             16.0
```

With 12-bit tables, the prefilter preserves the lower-cost probe variant too:

```text
method                         top1    cos_teacher  avg_summoned  avg_exact_scored  avg_candidates
sva_probe1_16x12               0.9106  0.9732       407.1         407.1             16.0
sva_probe1_prefilter32d128_16x12 0.9092 0.9707      407.1         128.0             16.0
```

Interpretation: the current best SVA shape is three-stage retrieval: summon broadly with adjacent buckets, use a cheap projected prefilter to reduce the pile, then run exact attention over the final candidates. This keeps most of the noisy-lookup robustness while making exact scoring much cheaper.

## Causal Sequence Checkpoint

I added `experiments/sva_causal_sequence_test.py`, which writes pages into an incremental causal cache and queries only prior pages.

At 1024 tokens with an average prefix of 512 pages, query noise 0.10, radius-1 probing, 12-bit addresses, and 24 tables:

```text
method                                  top1_target  cos_teacher  avg_summoned  avg_exact_scored
full_causal_attention                   0.9988       1.0000       512.0         512.0
sva_causal_probe1_24x12                 0.9954       0.9962       53.8          53.8
sva_causal_probe1_prefilter32d128_24x12 0.9957       0.9962       54.2          54.1
```

Interpretation: SVA transfers cleanly from static retrieval to an incremental causal-cache setting. The next risk is whether learned keys and queries preserve usable summon addresses in a tiny trainable model.

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
