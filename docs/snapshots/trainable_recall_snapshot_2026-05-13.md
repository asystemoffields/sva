# Trainable Recall Snapshot

Date: 2026-05-13

## Runs

### First H100 Run

The first H100 run used the original Python-loop SVA evaluation path.

Result:

- timed out after 3600 seconds
- no benchmark result returned
- live inspection showed one Python process burning about one CPU core
- H100 utilization sat around 40 percent with about 1.4GB memory used

Interpretation: this was a systems failure, not a scientific result. The SVA eval path was not GPU-native.

### Vectorized H100 Run

Commit: `de2b7b8`

Command:

```text
python -u experiments/sva_trainable_recall_test.py --steps 2500 --log-every 250 --batch-size 256 --eval-batch-size 64 --eval-batches 8 --n-pairs 32 --n-keys 128 --n-values 128 --d-model 128 --n-heads 4 --n-layers 2 --sva-tables 8 16 24 32 --sva-bits 10 --sva-budget 16 --sva-impl mask --probe-radius 1 --lr 4e-4
```

Result:

```text
train_done_seconds,94.62

method              loss    accuracy  avg_summoned  avg_verified
full_attention      4.4325  0.0371    0.0           0.0
sva_8x10            5.0829  0.0371    1.6           1.6
sva_probe1_8x10     4.6007  0.0352    5.1           5.1
sva_16x10           4.8271  0.0488    2.1           2.1
sva_probe1_16x10    4.4685  0.0410    7.7           7.5
sva_24x10           4.8206  0.0410    2.7           2.7
sva_probe1_24x10    4.4616  0.0410    9.8           9.1
sva_32x10           4.6317  0.0547    3.2           3.2
sva_probe1_32x10    4.4072  0.0391    11.3          10.0
```

Interpretation:

- The vectorized SVA eval path fixed the systems bottleneck. SVA eval went from timing out to sub-second per variant.
- The learned-representation test is inconclusive because the full-attention baseline did not learn the recall task well enough.
- SVA did not catastrophically break the weak model, but weak full-attention accuracy makes that evidence low value.

## Next Risk

The next H100 run should first make full attention solve the task. The most direct path is a curriculum or easier associative recall setup:

- fewer key-value pairs
- fewer key/value symbols
- deeper model or longer training
- checkpoint only when full attention is clearly above chance

After full attention learns, the SVA swap becomes meaningful.
