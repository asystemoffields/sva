# Late4 Static Tail-Buffered Inverted Decode Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Socket: late4 layers `26-29`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Decode mode: `inverted_static`
- Cells/subspace: `8`
- Budget: `128`
- Context: `32768`
- Full panel: 8 held-out keys x start/middle/end placements = 24 cases

## Runs

- Inner component profile: app `ap-66055onZ7dexjMDa0mwfhu`, function `fc-01KRMMFS9126FRS9XF801RQMAP`
- Outer component profile before tail buffering: app `ap-wyd8m7TnIk70PyPqlGHQRS`, function `fc-01KRMMV4M9VK3912Q4NCBTMQZK`
- Tail-buffer component profile: app `ap-Bfupu77mg21n7DnuaSWPJE`, function `fc-01KRMN3JV1NVNW5VH1MHXGNPD5`
- Tail-buffer full panel: app `ap-1wzep2GnkRTnc4fREIwf8c`, function `fc-01KRMN7EHVE184D2EGA03YVR0S`
- Device-ready buffer profile: app `ap-DhUSCbWQo7sYjIXoqsPkE7`, function `fc-01KRMNNS891SC8PTKESTFM22AV`
- Lazy product-code profile: app `ap-BkCPDQjfbk74c4iDrGeVUm`, function `fc-01KRMPSS8XH6TQRWD65QWWEWFS`
- Lazy product-code full panel: app `ap-Ak0GWLBYnqboJuT3pox4qs`, function `fc-01KRMPWY003JDNE5NE2PHANCZS`

## Result

The component profiles identified per-token key-catalog/posting maintenance as the largest decode cost. On the 6-case profiling slice before tail buffering, adapted SVA averaged:

- Static body: `1.299869 ms` per patched layer call
- Projection: `0.140379 ms`
- Key catalog/posting update: `3.363141 ms`
- Outer total: `4.955802 ms`
- Decode slowdown: `2.394732x`

Tail buffering keeps the static prompt postings fixed during decode, includes the generated tail as direct candidates, and rebuilds static postings every `64` generated tokens. On the same profiling slice after tail buffering, adapted SVA averaged:

- Static body: `1.199460 ms`
- Projection: `0.155428 ms`
- Key catalog/posting update: `2.095516 ms`
- Outer total: `3.611595 ms`
- Decode slowdown: `1.532751x`

On the full 24-case no-profile panel, tail-buffered adapted SVA reached:

- Answer NLL delta: `-0.427783`
- Answer KL to full: `0.038545`
- Top-1 agreement: `0.910714`
- Logit cosine: `0.992166`
- Decode slowdown: `1.836643x`
- Average verified tokens: about `126-127` on a nominal `128` budget

For comparison, the previous duplicate-refill static run at the same `8` cells/subspace had adapted KL `0.038739`, top-1 `0.910714`, cosine `0.992043`, NLL delta `-0.427092`, and decode slowdown `2.006075x`.

After this full panel, a small device-ready buffer follow-up moved SVA q/k projections, logit scale, and codebooks to float32 buffers on the model device at patch time. On the same 6-case profiling slice, adapted SVA changed from:

- Static body: `1.199460 ms` to `1.095723 ms`
- Projection: `0.155428 ms` to `0.118188 ms`
- Key catalog/posting update: `2.095516 ms` to `1.844932 ms`
- Outer total: `3.611595 ms` to `3.211059 ms`

Lazy product-code maintenance then stopped assigning product keys to each generated token on every decode step in `inverted_static` mode. Generated tokens remain visible through the direct tail candidate path, and the static posting catalog is rebuilt every `64` generated tokens. On the same 6-case profiling slice, adapted SVA averaged:

- Static body: `1.238693 ms`
- Projection: `0.104414 ms`
- Key catalog/posting update: `0.125164 ms`
- Outer total: `1.635716 ms`
- Decode slowdown: `1.289989x`

On the full 24-case no-profile panel, lazy product-code maintenance reached:

- Answer NLL delta: `-0.431091`
- Answer KL to full: `0.038447`
- Top-1 agreement: `0.916667`
- Logit cosine: `0.992165`
- Decode slowdown: `1.387741x`
- Prefill slowdown: `25.280631x`

## Interpretation

Tail buffering and lazy product-code maintenance are clean implementation wins: together they preserve the current quality target while cutting no-profile adapted decode slowdown from the duplicate-refill baseline `2.006075x` to `1.387741x`. The largest solved piece was per-token product-key/posting work in decode. The remaining wall-clock work is now mostly the static candidate/refill path and the exact/value gather path, while prefill remains much too expensive because it still builds and verifies SVA over the whole prompt.

Next target: push the fixed candidate path closer to kernel-shaped tensor operations, then attack prefill. The sharp decode test is whether static SVA can get below `1.0x` wall-clock at the same quality; the sharp long-context test is whether prefill/catalog construction can be amortized or chunked enough that SVA wins where full attention runs out of memory or context length.
