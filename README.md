# Summon-Verify Attention

Summon-Verify Attention (SVA) is a candidate sparse replacement for transformer attention.

The idea is simple:

1. Each page writes itself into several cheap content-addressed lookup tables.
2. A query activates the same addresses and summons a small candidate set.
3. A verifier runs exact dot-product attention over only the summoned candidates.

In the current toy tests, the write address and read address are the same object. That is the key design move: the memory does not need a separate librarian to learn where every page went.

## Quick Start

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python experiments\\sva_kill_test.py --task binding --trials 2 --tables 8 16 24 --bits 10 --budget 16 --query-noise 0.05 --logit-scale 16
```

## Current Result

On the 8192-page binding task with a 16-candidate verifier budget, SVA reaches near-full-attention top-1 recovery while reading only 16 candidates:

```text
method              top1    cos_teacher  avg_candidates
full_attention      1.0000  1.0000       8192.0
coarse_bank_verify  0.5552  0.5820       15.7
sva_16x10           0.9907  0.9721       16.0
sva_24x10           0.9995  0.9859       16.0
```

The newest robustness result is adjacent-bucket probing plus a cheap prefilter. Under noisy binding lookup, plain `sva_16x10` reached `top1=0.7124`; `sva_probe1_16x10` reached `top1=0.9263`, matching the full-attention teacher's `0.9287` on that setup while verifying 16 candidates.

With a 32-dimensional prefilter, `sva_probe1_prefilter32d128_16x10` kept `top1=0.9233` while reducing exact full-dimensional scoring from about 968 summoned pages to 128. That makes the working shape: summon broadly, cheap prefilter, exact verify.

The causal-cache test is now positive too. At 1024 tokens, `sva_causal_probe1_prefilter32d128_24x12` reached `top1=0.9957` with about 54 summoned prior pages on average, while full causal attention read about 512 prior pages on average in the same setup.

The pretrained socket test is now the sharpest signal. In `HuggingFaceTB/SmolLM2-135M-Instruct`, SVA can replace every Llama attention layer's score matrix while keeping the pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits. The best first H100 sweep matched full attention closely on short prompts:

```text
setting                         loss_delta  KL_to_full  top1_agree  logit_cos  avg_verified
32 tables, 10 bits, probe 2     0.093750    0.188110    0.783883    0.974020   20.432 / 53
```

The longer-context H100 sweep strengthened that result. At 512 tokens, `64 tables / 10 bits / probe 2 / budget 128` reached `loss_delta=0.015625`, `KL=0.009582`, `top1_agreement=0.970646`, and top-16 full-attention key recall `0.976191`.

The next research step is to make the summoned candidate set smaller. In the current socket harness, `avg_summoned` is the broad lookup set; without a prefilter, that is also the exact-scored set. At 512 tokens the strongest setting summons about 221 candidates before the post-score top-k budget.

The first random-projection prefilter reduced exact scoring but exposed the next bottleneck. At 256 tokens, `prefilter_dim=48 / prefilter_budget=64` cut exact scoring from about 113 candidates to 55 with `loss_delta=0.062500`. At 512 tokens, the same shape cut exact scoring from about 223 to 60 with `loss_delta=0.125000`. The next invention target is a better cheap ranker inside the summoned set.

The full-window real-QK address sweep now matches SmolLM2's configured context window: `max_position_embeddings=8192`, `seq_len=8192`. Random high-bit binary addresses are a kill for the million-token version of that exact address function. Aggregate top-16 recall at `14 bits / 128 tables / radius 2` was `0.838557`, but the random million-token candidate estimate was about `282k`. At `24 bits / 128 tables / radius 2`, the estimate falls to about `1.1k`, but top-16 recall was only `0.092231`.

The million-token pressure simulation sharpened that result using empirical hit density from real 8192-token SmolLM2 Q/K samples. The best aggregate recall was `20 bits / 256 tables / radius 2` at `0.384905`, but it projected to about `39.6k` average candidates at a million tokens, with p95 about `129k`. In the rough 128-1024 candidate band, aggregate recall stayed around 1-2%. The next work is a learned or model-aware address code.

The learned compressed-ranker test is the first strong follow-up. Training a small asymmetric Q/K ranker per layer/head on held-in query positions and evaluating held-out query positions reached aggregate top-16 recall `0.759781` with a 64-dimensional score and 256 verifier candidates, and `0.848338` with 512 verifier candidates. The next risk is held-out text generalization, then serving the learned score through an addressable lookup.

The held-out text test preserved the signal. Training on one 8192-token stream and evaluating on a reversed 8192-token stream reached aggregate top-16 recall `0.749752` with rank 64 and 256 verifier candidates, and `0.835488` with 512 verifier candidates. The next invention target is sublinear lookup for that learned compact score.

The first learned-score serving attempt tested random-hyperplane LSH over the rank-64 space. It is a kill for that specific lookup geometry. The strongest aggregate row reached only `0.233429` verified top-16 recall while projecting to about `38.6k` average candidates at a million-token context; in the rough few-hundred-candidate band, recall stayed around `0.013`.

Score-aware IVF routing improved the serving shape. Single-write k-means centroids over learned low-rank keys reached `0.234422` recall at about `3.5k` projected million-token candidates, and about `0.095-0.102` recall in the few-hundred-candidate band. That is much better than sign-LSH at the same scale, but still far below the learned ranker's all-key recall. The next target is multi-write or supervised routing: give each key more than one good way to be summoned, or train the catalog cells directly against top-key recall.

Multi-write IVF answered the first half of that branch. Giving each key `2,4,8` nearest-centroid writes modestly improved some local settings, but the best few-hundred to low-thousand candidate row reached `0.105422` recall at about `898` projected million-token candidates, close to single-write IVF's `0.102477` recall at about `783`. The highest-recall multi-write row reached `0.147647` at about `1,564` projected candidates, below single-write IVF's `0.166574` at about `1,666`. The current target is supervised routing or asymmetric compressed scoring: make the catalog optimize for top-key recall directly.

The first supervised query-cell router confirms that supervised routing can move recall, but the low-resolution cells are too dense. It reached `0.655816` aggregate recall at about `167k` projected million-token candidates, while the smallest setting reached only `0.039109` recall at about `3.7k` projected candidates. The next run raises the cell count to test whether the supervised signal survives in the `128-1024` projected-candidate band.

The high-resolution supervised router answered that directly. With `2048-4096` cells and small write/probe counts, it reached the projected candidate band but recall collapsed: `4096 cells / 4 writes / 2 probes` reached `0.012680` recall at about `999.5` projected candidates, and `2048 cells / 4 writes / 2 probes` reached `0.042721` at about `3.6k` projected candidates. The next branch is score-preserving compressed lookup, such as product-quantized or asymmetric scoring over the learned rank-64 keys.

Product-quantized learned-score lookup is the new best serving signal. The exact learned rank-64 scorer reached `0.754046` recall at a 256-candidate verifier budget and `0.839084` at 512. PQ with `16 subspaces / 256 codewords` reached `0.704985` at 256 and `0.803184` at 512, while `8 subspaces / 256 codewords` reached `0.647166` at 256 and `0.755937` at 512. This preserves most of the learned-ranker signal while replacing full low-rank key scoring with compact code lookups.

The first synthetic million-token throughput check is plausible but still costly if used everywhere. On H100 with stock PyTorch gather plus top-k, `8 x 256` PQ over 9 heads scanned one million keys in about `2.2 ms` for one query; `16 x 256` took about `4.5 ms`. The next target is coarse-to-fine PQ so the full scan can be cheaper and the high-quality score only runs on a shortlist.

Coarse-to-fine PQ preserved the fine-PQ signal. On held-out reversed 8192-token SmolLM2 streams, `4x64` coarse PQ shortlisting to `4096`, then `16x256` fine PQ and a `512` verifier budget, reached `0.799541` aggregate top-16 recall versus `0.801169` for full `16x256` fine-PQ scoring. With a `2048` shortlist it reached `0.789078`.

The staged path is also fast in the first direct million-token benchmark. On H100 with stock PyTorch gather plus top-k, `4x64 -> 16x256` coarse-to-fine PQ took about `1.91 ms` for one query and about `3.03 ms` for four queries over one million keys and 9 heads. That is faster than the previous full `8x256` scan while preserving nearly all of the full `16x256` fine-PQ recall at the larger shortlist. The next target is training the coarse stage against the fine-PQ winners so the `1024-2048` shortlist range can carry the same recall.

The first supervised coarse-stage attempt was a regression. Training a separate coarse ranker against the broad top-512 fine-PQ candidate set reached only `0.758293` aggregate recall at shortlist `4096`, below the unsupervised coarse-to-fine baseline's `0.803509`. The next test narrows the coarse target to attention top-16 labels.

The attention-label supervised coarse run recovered the signal and produced a small improvement. With attention top-16 labels, supervised `4x64` coarse PQ reached `0.802688` aggregate recall at shortlist `4096`, compared with `0.802021` for full fine PQ and `0.800967` for unsupervised `4x64` coarse-to-fine in the same run. At shortlist `2048`, it reached `0.797464` versus `0.792085` unsupervised. The result points toward sharper survival targets, while the next larger step is likely optimizing the coarse code inside the fine-ranker space instead of training a separate coarse ranker.

Attention-weighted coarse codebooks tested that next branch. Fitting the coarse codebook in the fine-ranker space with attention top-16 key boosts helped most at the tightest shortlist: `4x64` with boost `4` reached `0.773717` at shortlist `1024`, above `0.764865` unsupervised and above the separate supervised coarse ranker's `0.769128`. At `2048`, it reached `0.794999`, a smaller lift over unsupervised and below the separate supervised ranker's `0.797464`. At `4096`, it tied unsupervised. The next combined test is to train the separate coarse ranker, then fit attention-weighted codebooks inside that coarse space.

The combined test stacked those gains. Training the supervised rank-64 coarse scorer and then fitting attention-weighted `4x64` codebooks inside that coarse space reached `0.803184` at shortlist `4096`, `0.799882` at shortlist `2048`, and `0.776445` at shortlist `1024`. This is the current best serving candidate for the learned-ranker branch. The next pressure test is shorter shortlists, especially `512` and `768`, with the same verifier budget.

The tight-shortlist pressure test set the current practical band. Weighted supervised `4x64` reached `0.776445` at shortlist `1024`, `0.757239` at `768`, and `0.713759` at `512`. The method still improves over unweighted and unsupervised coarse PQ at each point, but the drop below `1024` is steep. The next target is a shortlist-aware coarse objective that trains for top-key survival at `512-1024` directly.

Hard-negative coarse training directly attacked that shortlist-survival objective. After `80` hard-negative steps, weighted hard-supervised `4x64` reached the strongest current mainline result. A mining-pool sweep found that pool `512` with boost `4` was best across aggregate shortlists: `0.827179` at shortlist `512`, `0.831303` at `768`, `0.834108` at `1024`, and `0.820685` at `2048`, versus exact learned-ranker recall of `0.839332` and full fine-PQ recall of `0.802021`.

The handoff diagnostic isolated the `2048` shortlist dip. On the same hard-negative candidate sets, coarse-only survival rose from `0.906095` at shortlist `1024` to `0.959279` at `2048`, and exact rank-64 rescoring stayed high at `0.847873` and `0.847811`. Fine-PQ rescoring fell from `0.834077` to `0.820669`. The next target is a cheap exact rank-64 middle stage: coarse PQ summon, exact low-rank rescore over roughly `1024-2048` candidates, then full attention verification over the final `512`.

The first synthetic million-token benchmark supports that target. With `4x64` coarse PQ over one million keys, exact rank-64 rescoring took about `1.00 ms` for one query and `2.16 ms` for four queries at shortlist `2048`, using about `1.15 GB` of bf16 rank-key memory for `9` heads. The exact rescore block itself was about `0.12 ms`; the measured cost is mostly coarse scan plus shortlist top-k. The next quality test is to socket this three-stage path into the SmolLM2 attention replacement harness.

The first three-stage socket result initially appeared to find a layer-specific failure mode, but that result was traced to a harness/interface bug. Artifact training was deriving Q/K routes from layer-boundary hidden states, while Hugging Face's Llama attention receives `input_layernorm(hidden_states)`. After applying the same input-layernorm before artifact Q/K extraction, the apparent fragile-layer behavior disappeared.

The corrected all-layer SmolLM2 socket result is now the strongest signal. At `seq_len=2048`, with `4x64` coarse PQ, `coarse_shortlist=1024`, and `budget=512`, replacing all 30 attention layers reached `loss_delta=0.000000`, `KL=0.000362`, `top1_agreement=0.994626`, `logit_cosine=0.997908`, and verified top-16 recall `0.999689` under teacher artifact training. Progressive artifact training was similarly strong: `KL=0.000361`, `top1_agreement=0.996092`, and verified top-16 recall `0.999703`.

The first frozen-artifact deployment proxy also passed at `seq_len=2048`: artifacts trained on the base calibration stream stayed effectively lossless on paragraph-order shifts (`rotate`, `reverse`, and `odds_evens`). That is a useful leakage audit.

The first full deployment benchmark is also positive. Artifacts frozen from 4096-token calibration documents were evaluated on held-out documents at 2048 and 4096 tokens. At `context=2048`, `coarse_shortlist=1024`, and `budget=512`, the all-layer socket reached `loss_delta=0.000000`, `KL=0.000165`, `top1_agreement=0.999145`, `logit_cosine=0.998157`, and verified top-16 recall `0.999083`. At `context=4096`, the same setting reached `loss_delta=0.000488`, `KL=0.000244`, `top1_agreement=0.999511`, `logit_cosine=0.990862`, and verified top-16 recall `0.995900`.

The full-window held-out benchmark preserves the signal at SmolLM2's configured `8192` token context. At `context=8192`, `coarse_shortlist=1024`, and `budget=512`, the all-layer socket reached `loss_delta=0.000732`, `KL=0.000564`, `top1_agreement=0.999756`, `logit_cosine=0.984127`, and verified top-16 recall `0.991471`. At `coarse_shortlist=2048` and `budget=512`, it reached `loss_delta=0.000794`, `KL=0.000481`, `top1_agreement=0.999786`, `logit_cosine=0.989240`, and verified top-16 recall `0.998707`.

The cached-decode benchmark splits quality from serving mechanics. With key-side low-rank catalogs and product codes precomputed once per layer, `context=8192`, `coarse_shortlist=2048`, and `budget=512` reached verified top-16 recall `0.998094`; `1024/512` reached `0.988018`. The current PyTorch lookup path is still slower than optimized full attention at 8192, around `3.1 ms` per decode lookup versus about `0.3 ms` for full attention in this harness, so the next systems frontier is a synthetic million-token cached decode test and then a fused/custom lookup path.

The synthetic million-token cached-decode benchmark found the first no-custom-kernel speed opening. Vectorized PyTorch SVA over one million keys took about `1.02 ms` for one decode query at `2048/512`, versus about `2.09 ms` for full attention. At four queries it was roughly parity, and at sixteen queries it lost (`7.13 ms` versus `3.44 ms`) because coarse score construction and top-k scale with the query batch. The next speed target is SVA itself: train for tighter `512-1024` shortlist survival and make the summon budget adaptive by layer, head, and query.

The tight-summon frontier shows that shortlist size is not the main no-kernel speed lever yet. Tightened artifact training at `context=8192` reached verified top-16 recall `0.962761` at `512/256`, `0.981497` at `768/256`, `0.989732` at `1024/256`, `0.995918` at `1536/256`, and `0.997870` at `2048/256`. At one million synthetic keys, vectorized SVA stayed around `0.96-0.97 ms` for one query across `512` through `2048` shortlists, while full attention was about `2.05-2.10 ms`. The cost is dominated by the full-cache coarse scan and top-k path, not the verifier.

Compact coarse-code sweeps found the first clear no-custom-kernel speed lever. At `context=8192`, `2x256` coarse codes reached verified top-16 recall `0.988336` at `1024/256`, `0.995258` at `1536/256`, and `0.998271` at `2048/512`, close to `4x64` while reducing one-query million-token latency from about `1.03 ms` to about `0.78 ms`. The faster `1x256` branch reached `0.995985` at `2048/512` and cut one-query latency to about `0.65 ms`; it also reduced `q=16` latency from `7.13 ms` to about `4.00 ms`.

The first deployable artifact bundle now exists locally at `results/hf_artifacts/sva-smollm2-135m-2x256-v1`. It contains all 30 layers for the `2x256` profile, with `bfloat16` low-rank projections and coarse codebooks, manifest metadata, default `2048/512` serving settings, and a small README. The bundle reloads through `experiments/sva_artifact_io.py` and is ready to publish as a Hugging Face artifact repo or GitHub release asset.

The first production-facing adapter now exists under `sva/`. It loads the artifact bundle, validates tensor shapes, reversibly patches Llama-family Hugging Face attention layers, records runtime stats, and reuses SVA key catalogs across cached decode steps. A local browser chat demo lives at `demo/local_chat_server.py`.

The 8k head-to-head result separates method quality from serving speed. At `context=8192`, `shortlist=2048`, and `budget=512`, the all-layer SVA socket reached `loss_delta=0.000907`, `KL=0.000446`, `top1_agreement=0.999695`, and `logit_cosine=0.991188`, with `16x` fewer exact scores and value reads. The current stock PyTorch adapter is slower wall-clock at 8k, so method-level wins have to come from longer contexts, richer catalogs, or a much tighter lookup implementation.

The first long-context extension proxy found the current boundary. With the frozen 8k `2x256` artifact, fixed `2048/512` recall falls from `0.977778` at 8k to `0.848611` at 32k, `0.657639` at 128k, and `0.365885` at 1M. Scaling the shortlist and verifier budget recovers the 128k case: `16384/2048` reached `0.943056` verified top-16 recall with `64x` fewer exact scores and value reads. At 1M, the same setting reached `0.645399`, which points to catalog capacity as the next method target.

The first language-level passkey benchmark is a sharper stress test. With the passkey at the beginning and the query at the end, adaptive inverted decode was too aggressive: at `4096` tokens it added `3.776614` answer NLL versus full attention while verifying about `32` tokens per decode query. Fixed scan decode with `2048/512` recovered the short-context behavior: answer NLL deltas were `0.076570` at `4096` and `0.163233` at `8192`, with `8x` and `16x` fewer value reads. Scaling to `8192/2048` recovered the longer rows too: answer NLL delta was `-0.016004` at `16384` and `0.116894` at `32768`, with `8x` and `16x` fewer value reads. Exact-string retrieval looks like a budget-policy problem through 32k, while the current prefill path remains the obvious systems bottleneck.

The first block-elevator benchmark tested a more kernel-shaped SVA path: summon contiguous blocks, then let selected blocks compute exact local softmax partials where they sit. Averaged over layers `0`, `15`, and `29`, token SVA with `2048` individual value reads fell to `output_cosine=0.943327` and `relative_error=0.456380` at `131072` synthetic tokens. Centroid block SVA with the same `2048` value reads reached `output_cosine=0.966790` and `relative_error=0.276183`, while reducing scattered segments from `2048` tokens to `32` contiguous blocks. At `8192`, token SVA remains stronger; at longer contexts, block statements look like a useful way to preserve diffuse value output.

The first token/block hybrid run found complementarity. With the same `2048` average value reads at `131072`, token SVA reached `output_cosine=0.944623` and `relative_error=0.457266`; block `64 x 32` reached `0.969223` and `0.270862`; an oracle selector between token and block reached `0.986559` and `0.165284` while reducing scattered segments from `2048` to about `853`. A cheap entropy selector improved exact-key survival over block-only but left much of the oracle gap open. The next target is a learned selector, then a hybrid serving-shaped passkey benchmark.

The learned selector is positive on held-out synthetic layer outputs. A tiny MLP trained on cheap pre-verifier features reached `train_accuracy=0.943673` and transferred to a different held-out document. At `131072`, learned `128 x 16` improved relative error from token SVA's `0.570985` and block-only `0.250723` to `0.179351`, with about `599` average contiguous/scattered segments instead of `2048` scattered token segments. At `32768`, learned `128 x 16` reached `relative_error=0.106382`, close to the oracle's `0.096472`. The next sharp test is language-facing: socket this dispatcher into the passkey benchmark and see whether it preserves exact retrieval while improving long-context diffuse output.

The evidence-haystack benchmark now measures summon quality directly. Multi-anchor summon improves evidence survival: at `8192` start placement, full-budget anchors lift summoned key survival from `0.592593` with one anchor to `0.962963` with eight anchors, and at `32768` end placement from `0.222222` to `0.777778` with sixteen anchors. The next bottleneck is verifier rerank: at `32768` end placement the key is summoned in `0.777778` of head/layer cases but survives final verification in `0.444444`. Split-budget anchors are useful when evidence is close to the query: `8192` end placement kept exact key survival at `1.000000` while verifying about `435` tokens, an `18.8x` read reduction.

The evidence-aware rerank sweep found a method-level improvement and a sharper next target. At `16384`, current-query rerank plus radius `32` lifted aggregate verified key survival from `0.461420` to `0.614198` while keeping average exact score work below full attention. At `32768`, expansion lifted aggregate candidate key coverage to `0.601852`, but verified key survival reached only `0.307098`; individual-token rerank is now the main loss point. The next test is span/block statements that preserve local evidence neighborhoods after summon.

The span-statement test confirms that local statements are a useful verifier shape, while also sharpening the summon problem. At `8192`, radius `32` improved aggregate output cosine from `0.991612` to `0.998145` and reduced scattered segments from about `273` to about `8`. At `16384`, statement-style verification still reached about `0.992-0.993` cosine in the best efficient rows. At `32768`, quality dropped into the `0.951-0.972` range and key survival stayed low, so the main pressure returns to catalog quality.

The rotation diagnostic found a large codebook-quality opening. Against the frozen artifact codebooks, aggregate teacher top-16 recall was `0.771888` at budget `512`, `0.837511` at `1024`, and `0.893808` at `2048`. Refit codebooks lifted those to `0.837637`, `0.884145`, and `0.923472`, while raising PQ score cosine from `0.870095` to about `0.9576`. Hadamard and signed-Hadamard refits were essentially tied with plain refit, so the next deployable test is held-out calibration-time codebook refresh rather than relying on eval-key refit.

Held-out codebook refresh confirmed that the codebook-quality opening generalizes. At `32768`, the frozen artifact reached teacher top-16 recall `0.563169/0.657407/0.752450` at budgets `512/1024/2048`. Calibration-fit codebooks lifted those to `0.635887/0.726002/0.809995`, close to the eval-refit upper bound `0.645553/0.734592/0.816090`. Code entropy moved with that win: normalized entropy rose from `0.718895` to about `0.978`, while the largest average code bucket fell from `0.230200` to about `0.0144`. At `8192`, the frozen artifact remains best, so the next deployable shape is context-matched profiles plus a simple context-length router.

The first long-context refreshed artifact is now exported at `results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1`. It reloads through the production artifact loader and reproduces the long-context refresh result: at `32768`, it reaches teacher top-16 recall `0.630588/0.725071/0.809860` at budgets `512/1024/2048`, with score cosine `0.945643` and normalized code entropy `0.978694`. At `8192`, it trails the original artifact, so the next production test is profile routing: original artifact for 8k, refreshed artifact for 16k/32k.

The first language-facing profile-router test is a useful negative. With the original profile at `8192`, the refreshed profile at `16384/32768`, and the larger `8192/2048` scan policy, passkey answer NLL deltas were `0.070947`, `0.042013`, and `0.138533`. The earlier original-profile scale-out row was slightly better at `16384/32768` (`-0.016004` and `0.116894`), so aggregate recall and entropy gains have not yet converted into passkey language gains. The next profile should be evidence-aware: weight codebook fit toward attention top-k and exact evidence neighborhoods, while recording entropy only as a collapse diagnostic.

Attention-weighted refresh is the first positive evidence-aware catalog result. At `32768`, plain refresh reached teacher top-16 recall `0.635887/0.726002/0.808793` at budgets `512/1024/2048`; strong attention-weighted refresh reached `0.677083/0.762297/0.836887`, also above the identity eval-refit ceiling `0.645526/0.734565/0.816081`. Entropy and score cosine moved down while teacher recall moved up, confirming that the objective is evidence survival, with entropy only a diagnostic.

The language-facing attention-weighted profile router gives a mixed result. With original profile at `8192`, strong attention-weighted profile at `16384/32768`, and scan `8192/2048`, passkey answer NLL deltas were `0.070947`, `0.024752`, and `0.152243`. This improves over the plain refreshed profile at `16384` (`0.042013` to `0.024752`) but regresses at `32768` (`0.138533` to `0.152243`). The next sharp move is a mixed-strength evidence-aware profile sweep: keep the evidence objective, reduce the blunt all-layer strong weighting, and record key survival beside answer NLL.

## Files

- `experiments/sva_kill_test.py`: standalone toy benchmark.
- `experiments/sva_causal_sequence_test.py`: incremental causal-cache benchmark.
- `experiments/sva_trainable_recall_test.py`: trainable modern-decoder recall benchmark.
- `experiments/sva_pretrained_socket_test.py`: pretrained SmolLM2 attention-socket benchmark.
- `experiments/sva_real_qk_address_sweep.py`: real-QK high-bit address sweep at the model's configured context window.
- `experiments/sva_million_stream_sim.py`: million-token address-pressure simulation from real SmolLM2 8192-token Q/K samples.
- `experiments/sva_learned_ranker_test.py`: learned compressed Q/K ranker test.
- `experiments/sva_learned_lsh_lookup_test.py`: learned-ranker random-hyperplane LSH serving test.
- `experiments/sva_learned_ivf_lookup_test.py`: learned-ranker IVF/centroid routing serving test.
- `experiments/sva_learned_multiwrite_ivf_lookup_test.py`: learned-ranker multi-write IVF serving test.
- `experiments/sva_supervised_query_router_test.py`: learned-ranker supervised query-cell router serving test.
- `experiments/sva_pq_lookup_test.py`: product-quantized learned-ranker lookup test.
- `experiments/sva_pq_scan_benchmark.py`: synthetic million-token PQ scan throughput benchmark.
- `experiments/sva_coarse_to_fine_pq_test.py`: coarse-to-fine product-quantized lookup test.
- `experiments/sva_coarse_to_fine_pq_scan_benchmark.py`: synthetic million-token coarse-to-fine PQ scan throughput benchmark.
- `experiments/sva_coarse_exact_rescore_benchmark.py`: synthetic million-token coarse PQ plus exact low-rank rescore benchmark.
- `experiments/sva_supervised_coarse_pq_test.py`: supervised coarse-stage PQ lookup test.
- `experiments/sva_deployment_socket_test.py`: frozen-artifact deployment proxy over simple text shifts.
- `experiments/sva_full_deployment_benchmark.py`: held-out-document deployment benchmark with context and budget sweeps.
- `experiments/sva_cached_decode_benchmark.py`: cached-key decode benchmark that separates lookup quality from full-socket harness overhead.
- `experiments/sva_million_cached_decode_benchmark.py`: synthetic million-token cached-decode benchmark comparing full attention with SVA lookup variants.
- `experiments/sva_8k_head_to_head_benchmark.py`: 8k full-attention versus SVA deployment benchmark with wall-clock and quality metrics.
- `experiments/sva_inverted_adaptive_decode_benchmark.py`: cached-decode benchmark for inverted-code adaptive SVA lookup.
- `experiments/sva_long_context_recall_sim.py`: long-context SVA recall proxy from real SmolLM2 Q/K activations and synthetic larger key banks.
- `experiments/sva_passkey_language_benchmark.py`: passkey-style long-context language benchmark scoring the correct answer tokens after cached prefill.
- `experiments/sva_block_elevator_benchmark.py`: block-first SVA benchmark that summons contiguous blocks and merges local softmax statements.
- `experiments/sva_block_hybrid_benchmark.py`: token/block hybrid benchmark that routes each head/query between scattered token SVA and contiguous block SVA.
- `experiments/sva_learned_hybrid_selector_benchmark.py`: learned selector benchmark for token/block SVA routing from cheap pre-verifier features.
- `experiments/sva_evidence_haystack_benchmark.py`: passkey evidence survival benchmark that measures whether the summoner keeps the needed tokens as context grows, with optional anchor-aware rerank and neighborhood expansion.
- `experiments/sva_span_statement_benchmark.py`: passkey span-statement benchmark that opens local spans around summoned evidence and compares selected-span output with full attention.
- `experiments/sva_rotation_diagnostic.py`: low-rank rotation diagnostic that compares frozen product codebooks with refit identity and Hadamard-style codebooks.
- `experiments/sva_codebook_refresh_benchmark.py`: held-out calibration-time codebook refresh benchmark for context-matched SVA catalogs.
- `experiments/sva_artifact_io.py`: save/load helpers for portable frozen SVA artifact bundles.
- `experiments/export_sva_artifact.py`: exporter for HF/GitHub-ready SVA artifact folders.
- `experiments/export_refreshed_sva_artifact.py`: exporter that refreshes artifact coarse codebooks on a calibration stream while preserving the trained low-rank projections.
- `experiments/sva_address_scaling.py`: address selectivity calculator for long contexts.
- `sva/`: production-facing artifact loader and Llama attention adapter.
- `demo/local_chat_server.py`: local HTML chat UI for SmolLM2 running with the exported SVA artifact.
- `modal_h100_trainable.py`: Modal H100 runner for the trainable benchmark.
- `modal_h100_socket.py`: Modal H100 runner for the pretrained socket sweep.
- `modal_h100_three_stage_socket.py`: Modal H100 runner for the three-stage pretrained socket test.
- `modal_h100_three_stage_socket_layers.py`: Modal H100 runner for layer-isolated three-stage socket tests.
- `modal_h100_three_condition_socket.py`: Modal H100 runner for hidden-state, progressive, and selective-hybrid socket conditions.
- `modal_h100_layer_frontier.py`: Modal H100 runner for selective socket layer-frontier tests.
- `modal_h100_layer_cliff.py`: Modal H100 runner for selective socket cliff-mapping tests.
- `modal_h100_layer_fallback.py`: Modal H100 runner for selective socket per-layer fallback tests.
- `modal_h100_layer_admission.py`: Modal H100 runner for automatic selective socket admission screening.
- `modal_h100_normfix_all_layers.py`: Modal H100 runner for all-layer socket tests after the attention-input normalization fix.
- `modal_h100_deployment_socket.py`: Modal H100 runner for the frozen-artifact deployment proxy.
- `modal_h100_full_deployment_benchmark.py`: Modal H100 runner for the held-out deployment benchmark.
- `modal_h100_full_deployment_8192.py`: Modal H100 runner for the full-window 8192 held-out deployment benchmark.
- `modal_h100_cached_decode_benchmark.py`: Modal H100 runner for the cached-key decode benchmark.
- `modal_h100_million_cached_decode.py`: Modal H100 runner for synthetic million-token cached-decode throughput.
- `modal_h100_tight_summon_frontier.py`: Modal H100 runner for the tight-shortlist quality/speed frontier.
- `modal_h100_compact_summon_frontier.py`: Modal H100 runner for compact coarse-code quality/speed frontier sweeps.
- `modal_h100_export_sva_artifact.py`: Modal H100 runner that exports the default `2x256` SVA artifact bundle to a Modal volume.
- `modal_h100_export_refreshed_artifact.py`: Modal H100 runner that exports a long-context calibration-refreshed artifact bundle.
- `modal_h100_export_attention_weighted_artifact.py`: Modal H100 runner that exports an attention-weighted long-context artifact bundle.
- `modal_h100_8k_head_to_head.py`: Modal H100 runner for the 8k head-to-head deployment benchmark.
- `modal_h100_inverted_adaptive_decode.py`: Modal H100 runner for adaptive inverted-code decode benchmarking.
- `modal_h100_inverted_posting_decode.py`: Modal H100 runner for cached posting-list decode benchmarking.
- `modal_h100_long_context_recall.py`: Modal H100 runner for the fixed-budget long-context recall proxy.
- `modal_h100_long_context_scaleout.py`: Modal H100 runner for the long-context shortlist and budget scale-out proxy.
- `modal_h100_passkey_language.py`: Modal H100 runner for adaptive inverted passkey language benchmarking.
- `modal_h100_passkey_language_scan.py`: Modal H100 runner for fixed-scan passkey language benchmarking.
- `modal_h100_passkey_language_scaleout.py`: Modal H100 runner for passkey shortlist and budget scale-out.
- `modal_h100_passkey_profile_router.py`: Modal H100 runner for passkey language tests with context-routed SVA profiles.
- `modal_h100_passkey_attention_weighted_router.py`: Modal H100 runner for passkey language tests with the attention-weighted long-context profile.
- `modal_h100_block_elevator.py`: Modal H100 runner for block-first SVA elevator benchmarking.
- `modal_h100_block_hybrid.py`: Modal H100 runner for token/block hybrid SVA benchmarking.
- `modal_h100_learned_hybrid_selector.py`: Modal H100 runner for learned token/block selector benchmarking.
- `modal_h100_evidence_haystack.py`: Modal H100 runner for passkey evidence survival benchmarking.
- `modal_h100_evidence_rerank.py`: Modal H100 runner for evidence-aware rerank and neighborhood-expansion benchmarking.
- `modal_h100_span_statement.py`: Modal H100 runner for passkey span-statement benchmarking.
- `modal_h100_rotation_diagnostic.py`: Modal H100 runner for low-rank rotation diagnostics.
- `modal_h100_codebook_refresh.py`: Modal H100 runner for held-out calibration-time codebook refresh.
- `modal_h100_attention_weighted_refresh.py`: Modal H100 runner for attention-weighted held-out codebook refresh.
- `modal_h100_refreshed_profile_recall.py`: Modal H100 runner for exported refreshed-profile recall sanity checks.
- `modal_h100_million_stream.py`: Modal H100 runner for the million-token address-pressure simulation.
- `modal_h100_learned_ranker.py`: Modal H100 runner for the learned compressed-ranker test.
- `modal_h100_learned_ranker_generalize.py`: Modal H100 runner for the held-out-text ranker test.
- `modal_h100_learned_lsh_lookup.py`: Modal H100 runner for learned-ranker LSH serving.
- `modal_h100_learned_ivf_lookup.py`: Modal H100 runner for learned-ranker IVF serving.
- `modal_h100_learned_multiwrite_ivf_lookup.py`: Modal H100 runner for learned-ranker multi-write IVF serving.
- `modal_h100_supervised_query_router.py`: Modal H100 runner for supervised query-cell router serving.
- `modal_h100_supervised_query_router_hires.py`: Modal H100 runner for high-resolution supervised query-cell router serving.
- `modal_h100_pq_lookup.py`: Modal H100 runner for product-quantized learned-ranker lookup.
- `modal_h100_pq_scan_benchmark.py`: Modal H100 runner for PQ scan throughput.
- `modal_h100_coarse_to_fine_pq.py`: Modal H100 runner for coarse-to-fine PQ lookup.
- `modal_h100_coarse_to_fine_pq_scan_benchmark.py`: Modal H100 runner for coarse-to-fine PQ scan throughput.
- `modal_h100_coarse_exact_rescore_benchmark.py`: Modal H100 runner for coarse PQ plus exact low-rank rescore throughput.
- `modal_h100_supervised_coarse_pq.py`: Modal H100 runner for supervised coarse-stage PQ lookup.
- `modal_h100_supervised_coarse_pq_attention16.py`: Modal H100 runner for supervised coarse PQ with attention top-16 labels.
- `modal_h100_weighted_coarse_pq.py`: Modal H100 runner for attention-weighted coarse PQ in the fine-ranker space.
- `modal_h100_weighted_supervised_coarse_pq.py`: Modal H100 runner for attention-weighted codebooks in a supervised coarse space.
- `modal_h100_weighted_supervised_coarse_pq_tight.py`: Modal H100 runner for tight-shortlist weighted supervised coarse PQ.
- `modal_h100_hard_supervised_coarse_pq.py`: Modal H100 runner for hard-negative supervised coarse PQ.
- `modal_h100_hard_pool_sweep.py`: Modal H100 runner for hard-negative pool-size sweep.
- `modal_h100_hard_handoff.py`: Modal H100 runner for hard-negative handoff diagnostics.
- `scripts/start_modal_h100_background.ps1`: detached Modal launcher that writes run logs under `results/modal_runs/`.
- `results/verification_snapshot_2026-05-13.md`: current kill-test results.
- `results/trainable_recall_snapshot_2026-05-13.md`: H100 trainable-representation checkpoint.
- `results/pretrained_socket_snapshot_2026-05-13.md`: SmolLM2 pretrained socket checkpoint.
- `results/pretrained_long_socket_snapshot_2026-05-13.md`: longer-context SmolLM2 socket checkpoint.
- `results/pretrained_prefilter_socket_snapshot_2026-05-13.md`: cheap-prefilter socket checkpoint.
- `results/real_qk_address_8192_snapshot_2026-05-13.md`: SmolLM2 full-window real-QK address sweep.
- `results/million_stream_snapshot_2026-05-13.md`: million-token address-pressure snapshot.
- `results/learned_ranker_snapshot_2026-05-13.md`: learned compressed-ranker snapshot.
- `results/learned_ranker_generalization_snapshot_2026-05-13.md`: held-out-text learned ranker snapshot.
- `results/learned_lsh_lookup_snapshot_2026-05-13.md`: learned-ranker LSH lookup snapshot.
- `results/learned_ivf_lookup_snapshot_2026-05-13.md`: learned-ranker IVF lookup snapshot.
- `results/learned_multiwrite_ivf_lookup_snapshot_2026-05-13.md`: learned-ranker multi-write IVF lookup snapshot.
- `results/supervised_query_router_snapshot_2026-05-13.md`: supervised query-cell router lookup snapshot.
- `results/supervised_query_router_hires_snapshot_2026-05-13.md`: high-resolution supervised query-cell router lookup snapshot.
- `results/pq_lookup_snapshot_2026-05-13.md`: product-quantized learned-ranker lookup snapshot.
- `results/pq_scan_benchmark_snapshot_2026-05-13.md`: synthetic million-token PQ scan throughput snapshot.
- `results/coarse_to_fine_pq_snapshot_2026-05-13.md`: coarse-to-fine PQ lookup snapshot.
- `results/coarse_to_fine_pq_scan_benchmark_snapshot_2026-05-13.md`: synthetic million-token coarse-to-fine PQ scan throughput snapshot.
- `results/coarse_exact_rescore_benchmark_snapshot_2026-05-13.md`: synthetic million-token coarse PQ plus exact low-rank rescore throughput snapshot.
- `results/supervised_coarse_pq_snapshot_2026-05-13.md`: supervised coarse-stage PQ lookup snapshot.
- `results/supervised_coarse_pq_attention16_snapshot_2026-05-13.md`: supervised coarse PQ with attention top-16 labels snapshot.
- `results/weighted_coarse_pq_snapshot_2026-05-13.md`: attention-weighted coarse PQ codebook snapshot.
- `results/weighted_supervised_coarse_pq_snapshot_2026-05-13.md`: weighted codebooks in supervised coarse space snapshot.
- `results/weighted_supervised_coarse_pq_tight_snapshot_2026-05-13.md`: tight-shortlist weighted supervised coarse PQ snapshot.
- `results/hard_supervised_coarse_pq_snapshot_2026-05-13.md`: hard-negative supervised coarse PQ snapshot.
- `results/hard_pool_sweep_snapshot_2026-05-13.md`: hard-negative mining-pool sweep snapshot.
- `results/hard_handoff_snapshot_2026-05-13.md`: hard-negative handoff diagnostic snapshot.
- `results/three_stage_socket_snapshot_2026-05-13.md`: three-stage socket and layer-isolation snapshot.
- `results/three_condition_socket_snapshot_2026-05-13.md`: hidden-state, progressive, and selective-hybrid socket comparison.
- `results/layer_frontier_snapshot_2026-05-13.md`: selective socket layer-frontier snapshot.
- `results/layer_cliff_snapshot_2026-05-13.md`: selective socket cliff-mapping snapshot.
- `results/layer_fallback_snapshot_2026-05-13.md`: selective socket per-layer fallback snapshot.
- `results/layer_admission_snapshot_2026-05-13.md`: automatic selective socket admission snapshot.
- `results/normfix_socket_audit_snapshot_2026-05-13.md`: attention-input normalization fix and corrected all-layer socket snapshot.
- `results/full_deployment_benchmark_snapshot_2026-05-14.md`: first held-out deployment benchmark snapshot.
- `results/full_deployment_8192_snapshot_2026-05-14.md`: full-window 8192 held-out deployment benchmark snapshot.
- `results/cached_decode_benchmark_snapshot_2026-05-14.md`: cached-key decode benchmark snapshot.
- `results/million_cached_decode_benchmark_snapshot_2026-05-14.md`: synthetic million-token cached-decode throughput snapshot.
- `results/tight_summon_frontier_snapshot_2026-05-14.md`: tight-shortlist quality and million-token speed frontier snapshot.
- `results/compact_summon_frontier_snapshot_2026-05-14.md`: compact coarse-code quality and million-token speed frontier snapshot.
- `results/artifact_export_snapshot_2026-05-14.md`: first local deployable SVA artifact export snapshot.
- `results/production_adapter_snapshot_2026-05-14.md`: first production-facing adapter and local chat demo snapshot.
- `results/long_context_extension_snapshot_2026-05-14.md`: 8k head-to-head plus 128k/1M long-context recall extension snapshot.
- `results/passkey_language_snapshot_2026-05-14.md`: first passkey-style language stress test for SVA decode policy.
- `results/block_elevator_snapshot_2026-05-14.md`: block-first SVA elevator and local statement benchmark snapshot.
- `results/block_hybrid_snapshot_2026-05-14.md`: token/block hybrid selector benchmark snapshot.
- `results/learned_hybrid_selector_snapshot_2026-05-14.md`: learned token/block selector benchmark snapshot.
- `results/evidence_haystack_snapshot_2026-05-14.md`: passkey evidence survival benchmark snapshot.
- `results/evidence_rerank_snapshot_2026-05-14.md`: evidence-aware rerank and neighborhood-expansion snapshot.
- `results/span_statement_snapshot_2026-05-14.md`: span-statement verifier benchmark snapshot.
- `results/rotation_diagnostic_snapshot_2026-05-14.md`: low-rank rotation and codebook-fit diagnostic snapshot.
- `results/codebook_refresh_snapshot_2026-05-14.md`: held-out calibration-time codebook refresh snapshot.
- `results/refreshed_profile_snapshot_2026-05-14.md`: exported long-context refreshed artifact and recall sanity snapshot.
- `results/passkey_profile_router_snapshot_2026-05-14.md`: language-facing passkey test for context-routed SVA profiles.
- `results/attention_weighted_refresh_snapshot_2026-05-14.md`: held-out attention-weighted refresh benchmark snapshot.
- `results/passkey_attention_weighted_router_snapshot_2026-05-14.md`: language-facing passkey test for the attention-weighted routed profile.
- `results/hf_artifacts/sva-smollm2-135m-2x256-v1/`: local HF/GitHub-ready `2x256` SVA artifact bundle.
- `results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1/`: local HF/GitHub-ready long-context refreshed `2x256` SVA artifact bundle.
- `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1/`: local HF/GitHub-ready attention-weighted long-context `2x256` SVA artifact bundle.
- `notes/attention_replacement_findings.md`: broader research log leading to SVA.
- `notes/hierarchical_tree_sva.md`: side-track notes for hierarchical chunk/tree SVA.
- `notes/million_token_scaling.md`: scaling target for million-token contexts.

## H100 Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-trainable
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-socket -ModalFile modal_h100_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-stage-socket -ModalFile modal_h100_three_stage_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-million-stream -ModalFile modal_h100_million_stream.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker -ModalFile modal_h100_learned_ranker.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker-generalize -ModalFile modal_h100_learned_ranker_generalize.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-lsh-lookup -ModalFile modal_h100_learned_lsh_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ivf-lookup -ModalFile modal_h100_learned_ivf_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-multiwrite-ivf-lookup -ModalFile modal_h100_learned_multiwrite_ivf_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-query-router -ModalFile modal_h100_supervised_query_router.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-query-router-hires -ModalFile modal_h100_supervised_query_router_hires.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-pq-lookup -ModalFile modal_h100_pq_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-pq-scan-benchmark -ModalFile modal_h100_pq_scan_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-to-fine-pq -ModalFile modal_h100_coarse_to_fine_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-to-fine-pq-scan-benchmark -ModalFile modal_h100_coarse_to_fine_pq_scan_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-exact-rescore-benchmark -ModalFile modal_h100_coarse_exact_rescore_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-coarse-pq -ModalFile modal_h100_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-coarse-pq-attention16 -ModalFile modal_h100_supervised_coarse_pq_attention16.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-coarse-pq -ModalFile modal_h100_weighted_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-supervised-coarse-pq -ModalFile modal_h100_weighted_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-supervised-coarse-pq-tight -ModalFile modal_h100_weighted_supervised_coarse_pq_tight.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-supervised-coarse-pq -ModalFile modal_h100_hard_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-pool-sweep -ModalFile modal_h100_hard_pool_sweep.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-handoff -ModalFile modal_h100_hard_handoff.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-stage-socket-layers -ModalFile modal_h100_three_stage_socket_layers.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-condition-socket -ModalFile modal_h100_three_condition_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-frontier -ModalFile modal_h100_layer_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-cliff -ModalFile modal_h100_layer_cliff.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-fallback -ModalFile modal_h100_layer_fallback.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-admission -ModalFile modal_h100_layer_admission.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-normfix-all-layers -ModalFile modal_h100_normfix_all_layers.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-deployment-socket -ModalFile modal_h100_deployment_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-full-deployment-benchmark -ModalFile modal_h100_full_deployment_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-full-deployment-8192 -ModalFile modal_h100_full_deployment_8192.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-cached-decode-benchmark -ModalFile modal_h100_cached_decode_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-million-cached-decode -ModalFile modal_h100_million_cached_decode.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-tight-summon-frontier -ModalFile modal_h100_tight_summon_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-compact-summon-frontier -ModalFile modal_h100_compact_summon_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-artifact -ModalFile modal_h100_export_sva_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-refreshed-artifact -ModalFile modal_h100_export_refreshed_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-block-elevator -ModalFile modal_h100_block_elevator.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-block-hybrid -ModalFile modal_h100_block_hybrid.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-hybrid-selector -ModalFile modal_h100_learned_hybrid_selector.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-evidence-haystack -ModalFile modal_h100_evidence_haystack.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-evidence-rerank -ModalFile modal_h100_evidence_rerank.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-profile-router -ModalFile modal_h100_passkey_profile_router.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-span-statement -ModalFile modal_h100_span_statement.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-rotation-diagnostic -ModalFile modal_h100_rotation_diagnostic.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-codebook-refresh -ModalFile modal_h100_codebook_refresh.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-attention-weighted-refresh -ModalFile modal_h100_attention_weighted_refresh.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-refreshed-profile-recall -ModalFile modal_h100_refreshed_profile_recall.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-attnweighted-artifact -ModalFile modal_h100_export_attention_weighted_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-attnweighted-router -ModalFile modal_h100_passkey_attention_weighted_router.py
```

The launcher uses `modal run --detach` and writes local metadata, stdout, stderr, and result files under `results/modal_runs/`.

Live progress is visible through Modal logs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```
