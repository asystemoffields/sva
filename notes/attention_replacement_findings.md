# Novel Attention Replacement Notes

Date: 2026-05-13

## Current Candidate

The strongest candidate from the first pass is **Address-Consistent Natural Delta Attention**.

The core idea is to replace one crowded KDA state with several recoverably addressed associative memories. Each write goes to a bank selected by an address function that the read path can reproduce. Inside each bank, the update is a preconditioned online-regression update.

```text
bank = address(key, optional_position_phase)

gain = P_bank k / (lambda + k^T P_bank k)
err  = v - M_bank k

M_bank <- M_bank + err gain^T
P_bank <- lambda^-1 (P_bank - gain k^T P_bank)

read(q):
  bank = address(q, optional_position_phase)
  return M_bank q
```

This differs from the first IRNDA sketch in one crucial way: the write route must be recoverable at read time. Value-aware write routing can store very well, and it needs an address receipt so the query can reliably find the bank.

## Prototype Result

I added a toy benchmark in `experiments/irnda_toy.py`. It writes random key-value bindings into bounded associative memories, then queries them later. The stressor is clustered keys, which creates finite-state collision.

Final 8-trial run:

```text
python experiments\irnda_toy.py --subkey-dim 64 --n-banks 4 --trials 8 --router-mode rls
```

Top-1 retrieval accuracy:

```text
clusters  delta   single_rls  hash_rls  irnda   irnda_oracle
384       0.198   0.567       0.948     0.543   0.998
96        0.167   0.530       0.936     0.495   0.987
32        0.091   0.512       0.926     0.446   0.970
16        0.051   0.514       0.932     0.425   0.943
8         0.027   0.521       0.932     0.421   0.947
4         0.012   0.529       0.911     0.439   0.942
```

Interpretation:

- The raw delta memory collapses as keys become more clustered.
- A single RLS-style memory is much stronger than raw delta.
- Consistent hash routing into RLS banks is dramatically stronger in this toy.
- The value-aware IRNDA write router has a high oracle ceiling, but the learned read router loses too much bank-address information.

The important finding is that banked memory works when write addressing and read addressing are tied together or when writes leave an explicit address receipt.

I then added a nonlinear route receipt using random Fourier features for the read router:

```text
python experiments\irnda_toy.py --subkey-dim 64 --n-banks 4 --trials 6 --router-mode rls --router-feature rff --route-dim 512
```

Top-1 retrieval accuracy:

```text
clusters  delta   single_rls  hash_rls  irnda_rff  irnda_oracle
384       0.195   0.576       0.947     0.846      0.998
96        0.170   0.531       0.944     0.754      0.990
32        0.092   0.513       0.920     0.698      0.970
16        0.053   0.520       0.934     0.661      0.941
8         0.028   0.522       0.924     0.634      0.946
4         0.012   0.535       0.913     0.605      0.942
```

The receipt code closes a large part of the read-router gap, but stable hash routing is still stronger in this benchmark. That suggests the best next mechanism should start with stable addressing, then use receipts only for high-risk overflow writes.

## Stable Overflow Receipts

I added **StableOverflowReceiptRLS**, which uses stable hash routing as the default path and only writes to an alternate bank when the primary bank looks much riskier than the best available bank.

Write path:

```text
primary_bank = stable_address(k)
best_bank = argmin_b risk(M_b, P_b, crowd_b, k, v)

if risk(primary_bank) > overflow_ratio * risk(best_bank):
  write best_bank
  write receipt: receipt_features(k) -> best_bank
else:
  write primary_bank
```

Read path:

```text
primary_bank = stable_address(q)
receipt_bank, confidence = receipt_memory(receipt_features(q))

if confidence is high:
  read receipt_bank
else:
  read primary_bank
```

The most useful setting so far is conservative:

```text
python experiments\irnda_toy.py --subkey-dim 64 --n-banks 4 --trials 6 --router-feature rff --route-dim 512 --overflow-ratio 20.0 --receipt-threshold 0.8
```

Top-1 retrieval accuracy:

```text
clusters  hash_rls  overflow  overflow_oracle
384       0.942     0.947     0.993
96        0.938     0.944     0.990
32        0.928     0.920     0.978
16        0.935     0.934     0.976
8         0.927     0.924     0.976
4         0.925     0.904     0.964
```

Interpretation:

- Conservative overflow adds a small gain when keys are less compressed.
- The oracle remains clearly above both hash routing and readable overflow, which means alternate-bank storage is useful.
- Dense clusters expose the next bottleneck: receipt calibration. Bad receipt overrides cost more than missed overflow opportunities.

Current research target:

```text
stable primary address
+ rare overflow writes
+ receipt confidence calibrated against primary-address trust
```

## Revised Mechanism

Use two address paths:

```text
primary_bank = stable_address(k)
risk = collision_pressure(primary_bank, k) + write_error(primary_bank, k, v)

if risk is low:
  write primary_bank
else:
  overflow_bank = least_interfering_bank(k, v)
  write overflow_bank
  write receipt: route_memory[k] -> overflow_bank
```

Read:

```text
primary_bank = stable_address(q)
receipt_bank = route_memory(q)

if receipt confidence is high:
  return M_receipt_bank q
else:
  return M_primary_bank q
```

The receipt memory can be tiny because it stores bank ids, not values. That gives the model a way to use value-aware interference routing without making the read path guess blindly.

## Why This Might Beat KDA

KDA uses one finite recurrent state per head. When many similar keys need different values, the state update must trade off old and new associations in the same address space.

Address-consistent banked memory adds a missing operation: allocation. It lets a head split its finite state into separately addressed regions, while the RLS/preconditioned update makes each region a better local associative learner.

This attacks three KDA weaknesses at once:

- collision: similar keys can be routed to different banks
- overwrite: high-confidence banks update more conservatively
- retrieval: reads use the same address path as writes

## Next Experiments

I added two kill-test modes:

- `--task overwrite`: the same key receives several values; the target is the latest value.
- `--task binding`: each entity has multiple attribute-specific facts, with keys composed as entity plus attribute.

Overwrite command:

```text
python experiments\irnda_toy.py --task overwrite --subkey-dim 64 --n-banks 4 --trials 3 --router-feature rff --route-dim 512 --overflow-ratio 20.0 --receipt-threshold 0.8 --forget 0.98
```

Top-1 retrieval accuracy:

```text
clusters  hash_rls  overflow  overflow_oracle  irnda  irnda_oracle
384       0.888     0.885     0.935            0.949  0.950
96        0.872     0.872     0.914            0.921  0.921
32        0.891     0.863     0.906            0.937  0.936
16        0.856     0.833     0.882            0.958  0.957
8         0.792     0.791     0.862            0.957  0.958
4         0.810     0.787     0.860            0.964  0.967
```

Binding command:

```text
python experiments\irnda_toy.py --task binding --n-items 128 --n-attributes 4 --subkey-dim 64 --n-banks 4 --trials 3 --cluster-counts 128 32 16 8 4 --router-feature rff --route-dim 512 --overflow-ratio 20.0 --receipt-threshold 0.8
```

Top-1 retrieval accuracy:

```text
clusters  hash_rls  overflow  overflow_oracle  irnda  irnda_oracle
128       0.639     0.591     0.619            0.449  0.902
32        0.562     0.606     0.649            0.423  0.856
16        0.586     0.581     0.624            0.332  0.817
8         0.672     0.604     0.634            0.346  0.801
4         0.587     0.548     0.590            0.331  0.763
```

I also tried primary-bank trust versus receipt-bank trust:

```text
use_receipt =
  receipt_route_confident
  and receipt_bank_trust > primary_bank_trust + margin
```

The simple trust estimate, `log1p(coverage) - log1p(uncertainty)`, did not materially change the collision or overwrite results. This says the current trust proxy is too weakly calibrated to solve receipt selection.

Current read:

- Stable hash routing is the strongest readable collision mechanism.
- Value-aware routing is excellent for overwrite when the read router can track it.
- Binding exposes a large readable/oracle gap for value-aware IRNDA.
- Conservative overflow is a partial patch, not the final mechanism.

## Top-K Recoverable Routing

I tried the more elegant version: constrain value-aware routing to a recoverable candidate set.

```text
candidates = top_k_address(key)
write_bank = argmax_b route_score_b(key) - risk_b(key, value)
```

Read uses the same candidate set:

```text
candidates = top_k_address(query)
output = weighted_read(candidates, route_score + bank_trust)
```

This is implemented as **TopKRecoverableRLS**.

First run:

```text
python experiments\irnda_toy.py --task collision --subkey-dim 64 --n-banks 4 --top-k 2 --trials 3 --router-feature rff --route-dim 512 --overflow-ratio 20.0 --receipt-threshold 0.8
```

Collision top-1:

```text
clusters  hash_rls  topk   topk_oracle
384       0.953     0.918  0.999
96        0.944     0.808  0.996
32        0.935     0.702  0.989
16        0.930     0.613  0.977
8         0.916     0.604  0.987
4         0.916     0.602  0.991
```

Overwrite top-1:

```text
clusters  hash_rls  topk   topk_oracle  irnda
384       0.877     0.826  0.937        0.949
96        0.895     0.790  0.933        0.921
32        0.846     0.796  0.930        0.937
16        0.792     0.792  0.944        0.958
8         0.792     0.776  0.936        0.957
4         0.677     0.709  0.883        0.964
```

Binding top-1 with tuned hard read/write:

```text
clusters  hash_rls  topk   topk_oracle  irnda_oracle
128       0.629     0.440  0.841        0.902
32        0.606     0.399  0.826        0.856
16        0.606     0.381  0.788        0.817
8         0.600     0.367  0.814        0.801
4         0.629     0.365  0.807        0.763
```

I also tried soft writes across the top-k candidate set. It reduced the oracle gap because reads no longer had to recover a single hidden winner, but it diluted storage. On overwrite, top-k soft write fell to roughly `0.68-0.77` top-1 while IRNDA stayed around `0.92-0.96`.

Interpretation:

- The top-k candidate set is a good storage constraint: `topk_oracle` is high.
- The read rule still cannot identify the useful candidate bank reliably.
- Softening the write makes readout easier but weakens the memory.
- The elegant recoverable-routing idea is alive, but the address function needs to be learned or trained against the oracle choices.

New design constraint:

```text
Do not hide a single value-aware winner inside top-k unless the address function
is trained to rank that winner at read time.
```

The next serious version should train the address scores using the oracle write bank as a target. That keeps the elegance: no external receipt table at inference, but the address function learns to make the chosen bank recoverable.

## Role-Matrix Membership Kill Test

I added a small isolated test in `experiments/role_matrix_kill_test.py`.

Question:

```text
Can a bank catalog recover which bank contains an (entity, attribute) binding
using role-separated matrix membership?
```

Compared sketches:

```text
flat scent:
  A_b += phi(entity + attr)
  score_b = A_b . phi(query)

symmetric lantern:
  S_b += z z^T
  score_b = z_q^T S_b z_q

role matrix:
  R_b += phi_entity(entity) phi_attr(attr)^T
  score_b = phi_entity(query_entity)^T R_b phi_attr(query_attr)
```

Default hash-routed run:

```text
python experiments\role_matrix_kill_test.py --route-mode hash --trials 8
```

Result:

```text
flat_top1  0.228
sym_top1   0.238
role_top1  0.237

flat_margin -2.250
sym_margin  -2.190
role_margin -2.088
```

More clustered entities:

```text
python experiments\role_matrix_kill_test.py --route-mode hash --n-entity-clusters 4 --trials 8
```

Result:

```text
flat_top1  0.279
sym_top1   0.281
role_top1  0.291

flat_margin -2.815
sym_margin  -2.727
role_margin -2.578
```

Balanced hash and larger role features:

```text
python experiments\role_matrix_kill_test.py --route-mode balanced_hash --role-dim 64 --trials 8
```

Result:

```text
flat_top1  0.178
sym_top1   0.189
role_top1  0.199

flat_margin -1.514
sym_margin  -1.465
role_margin -1.371
```

I also tried naive negative sketches for near-miss banks. They hurt.

Interpretation:

- Role-matrix scoring is directionally better than flat scent and symmetric lantern.
- The improvement is too small to solve the read-routing problem by itself.
- The failure mode is still bank crowding: every bank contains many entity-attribute relations, and raw additive sketches cannot separate them sharply enough.
- The “poetry” needs more than role separation. It needs either a trained scorer, sharper code construction, or a capacity/sparsity mechanism that prevents too many relational facts from collapsing into the same poem.

Verdict:

```text
Role-matrix membership: weak positive, insufficient as a standalone fix.
```

## Trained Nose Kill Test

I added `experiments/trained_nose_kill_test.py` to test a supervised lookup scorer.

Setup:

```text
freeze bank sketches
for each query, score every bank with a tiny MLP
train with cross-entropy over banks
features = [query, bank_vector, bank_matrix @ query, query * (bank_matrix @ query)]
```

Same frozen catalog, fresh noisy queries:

```text
python experiments\trained_nose_kill_test.py --epochs 60 --hidden-dim 256 --test-mode same_world
```

Result:

```text
metric      train   test
flat_top1   0.190   0.190
sym_top1    0.190   0.190
nose_top1   0.636   0.626
```

More clustered entities:

```text
python experiments\trained_nose_kill_test.py --n-entity-clusters 4 --epochs 60 --hidden-dim 256 --test-mode same_world
```

Result:

```text
metric      train   test
flat_top1   0.350   0.350
sym_top1    0.350   0.350
nose_top1   0.616   0.613
```

Larger catalog:

```text
python experiments\trained_nose_kill_test.py --n-entities 256 --epochs 60 --hidden-dim 256 --test-mode same_world
```

Result:

```text
metric      train   test
flat_top1   0.253   0.253
sym_top1    0.253   0.253
nose_top1   0.810   0.811
```

Fresh catalog split:

```text
python experiments\trained_nose_kill_test.py --epochs 60 --hidden-dim 256 --test-mode new_world
```

Result:

```text
metric      train   test
flat_top1   0.190   0.290
sym_top1    0.190   0.343
nose_top1   0.636   0.331
```

Interpretation:

- A trained query-conditioned scorer can learn to search a frozen catalog much better than hand scores.
- The gain survives fresh noisy queries against the same bank sketches.
- A scorer trained on one catalog does not automatically transfer to a fresh catalog of new facts.

Verdict:

```text
Trained nose: go, but only as an in-context/end-to-end lookup component.
```

For the full memory design, this means the scorer should be trained across many memory states, or trained online alongside the catalog, rather than fitted once to a single static sketch distribution.

## Summon-Verify Attention

Working name: **Summon-Verify Attention (SVA)**.

I added `experiments/self_summon_verifier_test.py` to test the two-stage idea:

```text
1. pages write themselves into several content-addressed LSH tables
2. query activates matching buckets
3. verifier performs exact dot-product attention over only the summoned candidates
```

This asks whether we can approximate an attention field without a learned librarian. The page writes and query lookup use the same addresses.

Stress setting:

```text
4096 stored pages
1024 queries
budget: verifier can inspect only 16 candidates
query noise: 0.05
teacher attention logit scale: 16
```

Random clustered keys:

```text
python experiments\self_summon_verifier_test.py --task random --trials 3 --tables 4 8 16 --bits 10 --budget 16 --query-noise 0.05 --logit-scale 16
```

Result:

```text
method             top1   cos_teacher  avg_candidates
full_attention     1.000  1.000        4096
coarse_bank        0.534  0.547        15
self_summon_4x10   0.735  0.742        16
self_summon_8x10   0.917  0.914        16
self_summon_16x10  0.995  0.987        16
```

Entity-attribute binding keys:

```text
python experiments\self_summon_verifier_test.py --task binding --trials 3 --tables 4 8 16 --bits 10 --budget 16 --query-noise 0.05 --logit-scale 16
```

Result:

```text
method             top1   cos_teacher  avg_candidates
full_attention     1.000  1.000        4096
coarse_bank        0.505  0.537        15
self_summon_4x10   0.741  0.749        16
self_summon_8x10   0.933  0.918        16
self_summon_16x10  0.993  0.983        16
```

Interpretation:

- This is the strongest result so far.
- Self-summoning lookup has much higher recall than coarse bank routing at the same verifier budget.
- The verifier does not need to be smart when retrieval has high recall; exact dot-product over the summoned set is enough.
- The write/read address contract is naturally solved: pages write to the same addresses that queries activate.

Verdict:

```text
Summon-Verify Attention: strong go.
```

The next architecture should probably shift from “one learned librarian finds the room” to:

```text
many cheap self-indexing traces summon candidates
+ small verifier/reranker performs precise attention over the summoned set
```

Next:

1. Keep **recoverable allocation + preconditioned update** as the main thesis.
2. Train the recoverable address function against oracle write-bank targets.
3. Compare equal-state-budget variants by shrinking bank width.
4. Move from random vectors to a tiny PyTorch sequence model with learned keys and values.

The key falsification test: receipt routing should close most of the gap between readable overflow and `overflow_oracle` on dense clusters.

## SmolLM2 Socket Frontier

The current SVA evidence has moved from toy lookup to pretrained model surgery. In `HuggingFaceTB/SmolLM2-135M-Instruct`, the strongest path is selective SVA socketing: keep the pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits, and replace selected attention score matrices with the three-stage SVA path.

Latest frontier result:

```text
socketed layers                         loss_delta  KL        top1_agree  verified_top16_recall
0,1,3,4,7,10                            0.007812    0.008817  0.994138    0.874180
0,1,3,4,5,6,7,8,10                      0.011719    0.010025  0.992672    0.850978
0,1,3,4,5,6,7,8,9,10,15,18              0.015625    0.014993  0.992672    0.828815
0,1,3,4,5,6,7,8,9,10,13,15,17,18,21     0.019531    0.018851  0.991695    0.810719
0,1,3,4,5,6,7,8,9,10,13,14,15,16,17,
18,19,20,21,23                          0.757812    0.757605  0.860772    0.779792
```

Readout:

- The frontier is smooth through 15 of 30 layers.
- The 20-layer set breaks at the model level more than local recall alone predicts.
- Layers `14`, `16`, `19`, `20`, and `21` are immediate suspects from the failed frontier's worst-head diagnostics.

Next falsification test: add suspect layers one at a time to the 15-layer set, then repeat the bad additions under progressive artifact training. The theory should now be shaped around the 15-to-20 cliff.

Cliff-map result:

```text
condition                  training      loss_delta  KL        top1_agree  verified_top16_recall
base_15                    teacher       0.023438    0.024391  0.988764    0.811023
add_14                     teacher       0.027344    0.028440  0.986810    0.807540
add_16                     teacher       0.078125    0.078464  0.974597    0.799713
add_19                     teacher       0.562500    0.562174  0.901808    0.797899
add_20                     teacher       0.023438    0.024428  0.989253    0.810775
add_23                     teacher       0.019531    0.018521  0.990718    0.813999
add_14_16                  teacher       0.082031    0.081203  0.973131    0.796423
add_19_20                  teacher       0.468750    0.463211  0.917929    0.793249
frontier_20                teacher       0.757812    0.751164  0.862726    0.779580
add_14_16                  progressive   0.066406    0.065821  0.982413    0.793548
add_19_20                  progressive   0.128906    0.127494  0.977528    0.789265
frontier_20                progressive   0.179688    0.179911  0.963850    0.781464
```

Layer `19` is the primary fault line. Progressive training helps a lot but leaves a quality gap. The next direct test is a per-layer fallback: socket the 20-layer frontier except layer `19`, and compare teacher versus progressive artifacts.

Per-layer fallback result:

```text
condition             training      socketed_layers  loss_delta  KL        top1_agree  verified_top16_recall
base_15               teacher       15               0.019531    0.018087  0.991207    0.810951
add_19                progressive   16               0.156250    0.154671  0.977040    0.804819
no_19                 teacher       19               0.097656    0.097367  0.967758    0.798275
no_19                 progressive   19               0.074219    0.071809  0.977528    0.796193
no_16_19              teacher       18               0.042969    0.042852  0.984367    0.809804
no_16_19              progressive   18               0.039062    0.040822  0.985344    0.810279
```

The current best larger socket is 18 layers, keeping `16` and `19` as full attention. For a much larger model, this needs to become an automatic layer admission rule that measures downstream distribution preservation, not a hand-curated exception list.

First admission-screen result:

```text
baseline socket set: 0,1,3,4,5,6,7,8,9,10,13,15,17,18,21

admit:  2,12,14,20,22,23,24,25,26,27,28,29
reject: 11,16,19

final admitted socket set:
0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,20,21,22,23,24,25,26,27,28,29

final teacher:     loss_delta=0.117188, KL=0.114144, top1=0.971666, verified_top16=0.750432
final progressive: loss_delta=0.054688, KL=0.054555, top1=0.983390, verified_top16=0.786364
```

Caveat: this screened candidates against the base 15-layer set, then validated the combined admitted set. The next compiler-style version should test each candidate against the currently admitted set.

The mechanism question was resolved as a harness/interface bug. Artifact training was deriving Q/K from layer-boundary hidden states, but Llama attention receives `input_layernorm(hidden_states)`. After applying the same input-layernorm before artifact Q/K extraction, the rejected-layer pattern disappeared.

Norm-fixed cliff result:

```text
condition                  training      loss_delta  KL        top1_agree  verified_top16_recall
base_15                    teacher       0.000000    0.000354  0.993161    0.999852
add_16                     teacher       0.000000    0.000347  0.994138    0.999859
add_19                     teacher       0.000000    0.000340  0.995115    0.999847
frontier_20                teacher       0.000000    0.000355  0.995115    0.999799
frontier_20                progressive   0.000000    0.000353  0.992672    0.999817
```

Norm-fixed all-layer result:

```text
condition        training      socketed_layers  loss_delta  KL        top1_agree  verified_top16_recall
all layers       teacher       30               0.000000    0.000362  0.994626    0.999689
all layers       progressive   30               0.000000    0.000361  0.996092    0.999703
```

The prior selective-layer, cliff, fallback, and admission snapshots are useful as a debugging trail, but their layer-fragility interpretation is stale. The corrected next target is the scale frontier: longer contexts, smaller shortlists and verifier budgets, and normalized-Q/K million-token simulations.
