# Side Experiment: Full-Layer SVA Fine-Tune on 20M Tokens

## Question

Can a small SmolLM2 model adapt when SVA replaces attention across every layer, and can it learn to use the SVA bottleneck in a way that improves quality or unlocks lower budgets?

This is a side branch beside the current late4 production path. The result we want is simple: after training with SVA active, the model should close the quality gap to full attention at materially lower value-read budgets, especially at contexts where full attention becomes expensive.

## Core Run

- Base model: `HuggingFaceTB/SmolLM2-135M`.
- Token budget: `20M` unique training tokens.
- Epochs: `3`, for `60M` token exposures.
- Context lengths: train mostly at `2048` or `4096`, then evaluate at `8192`, `32768`, and longer passkey contexts where the SVA artifacts support it.
- Replacement: SVA sockets all Llama attention layers from step 0.
- Initial SVA artifact: current attention-weighted `2x256` artifact, then a refreshed/context-matched profile if the first pilot says summon recall is the bottleneck.
- Trainable parameters, first run: base model weights plus small SVA residual output adapters.
- Frozen parameters, first run: SVA codebooks, candidate selection, and exact verifier policy.

The frozen summon policy keeps this run interpretable: if the model adapts, we learn that training under the SVA routing distribution matters. Later runs can train the summon machinery directly.

## Controls

Use the same data order, token count, optimizer, and batch shape for:

- full-attention continued pretrain control;
- all-layer SVA continued pretrain;
- late4 SVA adapter control, if compute allows.

The full-attention control tells us whether the data recipe itself helped or hurt the base model. The late4 control tells us whether the full-layer SVA run is discovering something beyond the already-strong late-layer socket.

## Pilot Before The Full Run

Start with a `1M` token, `1` epoch pilot using the exact harness planned for the full run.

Pilot pass condition:

- held-out LM loss moves in the same direction as training loss;
- passkey answer KL/top-1 improves versus frozen all-layer SVA;
- no layer or head produces repeated runtime failures;
- the saved checkpoint can reload into the local chat/demo path.

If the pilot passes, launch the `20M x 3` run.

## Training Objective

Primary objective:

- standard next-token cross entropy with SVA active.

Optional stabilizer for the first epoch:

- KL to full-attention teacher logits on a small anchor batch, sampled every `N` optimizer steps.

The anchor KL should be measured and logged even if its weight is zero. It gives us a direct drift meter against the model we are trying to improve.

## Metrics

Track during training:

- train loss and held-out loss;
- teacher-logit KL, top-1 agreement, and logit cosine on anchor batches;
- per-layer adapter update norms;
- summon stats: average summoned, verified, cell visits, and value reads;
- wall-clock tokens/sec and peak memory.

Evaluate after each epoch:

- held-out perplexity at normal context;
- 8K head-to-head against full attention;
- 32K passkey answer-decode: answer NLL delta, KL, top-1, logit cosine;
- long-context recall proxy at `32K`, `128K`, and `1M` if artifact capacity supports it;
- local chat smoke test with the SVA socket loaded.

## Initial Hyperparameters

- Sequence length: `2048` for pilot; `4096` if H100 memory leaves comfortable margin.
- Global batch: target `64K` tokens per optimizer step via gradient accumulation.
- Optimizer: AdamW.
- Learning rate: `1e-5` for full model weights, `1e-3` for residual adapters.
- Weight decay: `0.1` for model weights, `0.0` for residual adapters.
- Warmup: `3%` of steps.
- Scheduler: cosine to `10%` of peak LR.
- Precision: bf16.
- Checkpoints: every `0.5` epoch plus final.

At `64K` tokens per optimizer step, `60M` token exposures is about `938` optimizer steps.

## Implementation Notes

Current `SVALlamaAttention` is built as an inference module, so the first harness should keep the model in eval mode while enabling gradients on model weights and adapters. That preserves SVA behavior while allowing backprop through the selected value path and residual stream. Candidate selection/top-k remains a discrete routing decision for this branch.

The clean harness shape is:

1. Load SmolLM2 and tokenizer.
2. Patch every attention layer with SVA.
3. Wrap each SVA layer with the same residual output adapter used in the late4 distillation run.
4. Enable gradients for base weights and adapters.
5. Train CE on causal LM batches with SVA active.
6. Save checkpoint as base-model delta plus SVA adapter weights and artifact manifest.
7. Evaluate with the same answer-decode and long-context harnesses used by the main path.

## Decision Rule

Treat this as a go if the full-layer SVA model recovers most of the frozen all-layer quality gap and improves long-context or budget behavior beyond late4.

Treat it as a side result if it mainly proves that SVA-active fine-tuning helps but late4 still dominates the production-shaped path.
