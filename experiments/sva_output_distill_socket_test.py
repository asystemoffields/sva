"""Output-distillation probe for socketed SVA layers.

This test asks whether early/mid-layer SVA drift is correctable when the model
is trained in the state distribution created by SVA itself. The base model and
SVA artifacts stay frozen; only tiny residual adapters after SVA attention are
trained to match the full-attention teacher logits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_deployment_socket_test import make_text
from sva_pretrained_socket_test import (
    ThreeStageLayerArtifacts,
    build_artifacts_for_hidden_states,
    build_progressive_three_stage_artifacts,
    compare_logits,
    config_from_args,
    encode_batch,
    format_layer_list,
    parse_layer_list,
    patch_model_with_sva,
    shifted_loss,
)


@dataclass
class CachedBatch:
    mode: str
    batch: dict[str, torch.Tensor]
    teacher_logits: torch.Tensor


class SVAResidualOutputAdapter(nn.Module):
    def __init__(
        self,
        sva_attn: nn.Module,
        hidden_size: int,
        rank: int,
        source: str,
        scale: float,
    ) -> None:
        super().__init__()
        if source not in {"attn", "input", "both"}:
            raise ValueError(f"Unknown adapter source: {source}")
        self.sva_attn = sva_attn
        self.source = source
        self.scale = scale
        in_features = hidden_size * 2 if source == "both" else hidden_size
        self.down = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        attn_output, attn_weights = self.sva_attn(hidden_states, *args, **kwargs)
        if self.source == "attn":
            adapter_input = attn_output.float()
        elif self.source == "input":
            adapter_input = hidden_states.float()
        else:
            adapter_input = torch.cat([hidden_states.float(), attn_output.float()], dim=-1)
        correction = self.up(F.gelu(self.down(adapter_input))) * self.scale
        return attn_output + correction.to(dtype=attn_output.dtype), attn_weights


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise ValueError("At least one text mode is required.")
    return modes


def reset_sva_stats(cfg) -> None:
    cfg.stats.clear()
    cfg.layer_stats.clear()
    cfg.head_stats.clear()


def cache_teacher_outputs(
    model: nn.Module,
    tokenizer,
    modes: list[str],
    repeats: int,
    max_length: int,
    device: torch.device,
) -> list[CachedBatch]:
    cached: list[CachedBatch] = []
    model.eval()
    with torch.no_grad():
        for mode in modes:
            text = make_text(mode, repeats)
            batch = encode_batch(tokenizer, [text], max_length, device)
            output = model(**batch, use_cache=False)
            cached.append(CachedBatch(mode=mode, batch=batch, teacher_logits=output.logits.detach()))
    return cached


def distill_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    valid = attention_mask[:, 1:].bool()
    teacher = teacher_logits[:, :-1, :][valid].float() / temperature
    student = student_logits[:, :-1, :][valid].float() / temperature
    teacher_probs = F.softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def evaluate_split(
    model: nn.Module,
    cached_batches: list[CachedBatch],
    phase: str,
    cfg,
) -> list[dict[str, float | str]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for cached in cached_batches:
            reset_sva_stats(cfg)
            student_logits = model(**cached.batch, use_cache=False).logits
            teacher_loss = shifted_loss(
                cached.teacher_logits,
                cached.batch["input_ids"],
                cached.batch["attention_mask"],
            )
            student_loss = shifted_loss(
                student_logits,
                cached.batch["input_ids"],
                cached.batch["attention_mask"],
            )
            metrics = compare_logits(cached.teacher_logits, student_logits, cached.batch["attention_mask"])
            rows.append(
                {
                    "phase": phase,
                    "mode": cached.mode,
                    "full_loss": float(teacher_loss.item()),
                    "student_loss": float(student_loss.item()),
                    "loss_delta": float((student_loss - teacher_loss).item()),
                    "kl_to_full": metrics["kl_to_full"],
                    "top1_agreement": metrics["top1_agreement"],
                    "logit_cosine": metrics["logit_cosine"],
                    "avg_summoned": float(cfg.stats["summoned"] / max(cfg.stats["queries"], 1.0)),
                    "avg_verified": float(cfg.stats["verified"] / max(cfg.stats["queries"], 1.0)),
                }
            )
    return rows


def wrap_socket_layers_with_adapters(
    model: nn.Module,
    socket_layers: list[int] | None,
    rank: int,
    source: str,
    scale: float,
) -> None:
    selected = None if socket_layers is None else set(socket_layers)
    hidden_size = int(model.config.hidden_size)
    for layer_idx, layer in enumerate(model.model.layers):
        if selected is None or layer_idx in selected:
            device = next(layer.self_attn.parameters()).device
            layer.self_attn = SVAResidualOutputAdapter(layer.self_attn, hidden_size, rank, source, scale).to(device)


def print_rows(rows: list[dict[str, float | str]]) -> None:
    print(
        "result_header,phase,mode,full_loss,student_loss,loss_delta,"
        "kl_to_full,top1_agreement,logit_cosine,avg_summoned,avg_verified"
    )
    for row in rows:
        print(
            "result,"
            f"{row['phase']},{row['mode']},"
            f"{row['full_loss']:.6f},{row['student_loss']:.6f},{row['loss_delta']:.6f},"
            f"{row['kl_to_full']:.6f},{row['top1_agreement']:.6f},{row['logit_cosine']:.6f},"
            f"{row['avg_summoned']:.3f},{row['avg_verified']:.3f}"
        )


def print_mean_rows(rows: list[dict[str, float | str]]) -> None:
    by_phase: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_phase.setdefault(str(row["phase"]), []).append(row)
    print("mean_header,phase,loss_delta,kl_to_full,top1_agreement,logit_cosine")
    for phase, phase_rows in by_phase.items():
        loss_delta = sum(float(row["loss_delta"]) for row in phase_rows) / len(phase_rows)
        kl = sum(float(row["kl_to_full"]) for row in phase_rows) / len(phase_rows)
        top1 = sum(float(row["top1_agreement"]) for row in phase_rows) / len(phase_rows)
        cosine = sum(float(row["logit_cosine"]) for row in phase_rows) / len(phase_rows)
        print(f"mean,{phase},{loss_delta:.6f},{kl:.6f},{top1:.6f},{cosine:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Output-distill tiny adapters on top of socketed SVA.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-text-mode", choices=["base", "rotate", "reverse", "odds_evens"], default="base")
    parser.add_argument("--adapter-train-text-modes", default="base,reverse")
    parser.add_argument("--eval-text-modes", default="rotate,odds_evens")
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--socket-layers",
        default="0-25",
        help="Comma-separated layers or ranges to replace. Empty means all layers.",
    )
    parser.add_argument("--route-source", choices=["qk", "hidden"], default="qk")
    parser.add_argument("--artifact-training", choices=["teacher", "progressive"], default="teacher")
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prefilter-dim", type=int, default=0)
    parser.add_argument("--prefilter-budget", type=int, default=0)
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=4)
    parser.add_argument("--coarse-codewords", type=int, default=64)
    parser.add_argument("--coarse-shortlist", type=int, default=1024)
    parser.add_argument("--coarse-label-topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-train-steps", type=int, default=160)
    parser.add_argument("--coarse-hard-steps", type=int, default=80)
    parser.add_argument("--coarse-hard-pool", type=int, default=512)
    parser.add_argument("--coarse-hard-negatives", type=int, default=64)
    parser.add_argument("--coarse-hard-margin", type=float, default=1.0)
    parser.add_argument("--coarse-hard-lr-scale", type=float, default=0.5)
    parser.add_argument("--weighted-boost", type=float, default=4.0)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--ranker-lr", type=float, default=0.003)
    parser.add_argument("--ranker-weight-decay", type=float, default=0.0001)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--diagnose-topk", type=int, default=0)
    parser.add_argument("--head-report-limit", type=int, default=0)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--adapter-source", choices=["attn", "input", "both"], default="both")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--distill-steps", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=0.001)
    parser.add_argument("--adapter-weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()
    args.mode = "three_stage"

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dtype_map = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    artifact_batch = encode_batch(
        tokenizer,
        [make_text(args.artifact_text_mode, args.text_repeats)],
        args.max_length,
        device,
    )
    train_modes = parse_modes(args.adapter_train_text_modes)
    eval_modes = parse_modes(args.eval_text_modes)
    cache_modes = []
    for mode in train_modes + eval_modes:
        if mode not in cache_modes:
            cache_modes.append(mode)
    cached = cache_teacher_outputs(model, tokenizer, cache_modes, args.text_repeats, args.max_length, device)
    train_cached = [item for item in cached if item.mode in train_modes]
    eval_cached = [item for item in cached if item.mode in eval_modes]

    if args.artifact_training == "progressive":
        artifacts = build_progressive_three_stage_artifacts(model, artifact_batch, socket_layers, args, device)
    else:
        with torch.no_grad():
            artifact_output = model(**artifact_batch, use_cache=False, output_hidden_states=True)
        if artifact_output.hidden_states is None:
            raise ValueError("Artifact training requires hidden states.")
        artifacts: dict[int, ThreeStageLayerArtifacts] = build_artifacts_for_hidden_states(
            model,
            artifact_output.hidden_states,
            socket_layers,
            args,
            device,
        )
        del artifact_output

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cfg = config_from_args(args, artifacts)
    patch_model_with_sva(model, cfg, socket_layers)
    wrap_socket_layers_with_adapters(model, socket_layers, args.adapter_rank, args.adapter_source, args.adapter_scale)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.adapter_lr,
        weight_decay=args.adapter_weight_decay,
    )

    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"artifact_text_mode,{args.artifact_text_mode}")
    print(f"adapter_train_text_modes,{';'.join(train_modes)}")
    print(f"eval_text_modes,{';'.join(eval_modes)}")
    print(f"text_repeats,{args.text_repeats}")
    print(f"seq_len,{artifact_batch['input_ids'].shape[1]}")
    print(f"socket_layers,{format_layer_list(socket_layers)}")
    print(f"socket_layer_count,{len(socket_layers) if socket_layers is not None else len(model.model.layers)}")
    print(f"route_source,{args.route_source}")
    print(f"artifact_training,{args.artifact_training}")
    print(f"adapter_rank,{args.adapter_rank}")
    print(f"adapter_source,{args.adapter_source}")
    print(f"distill_steps,{args.distill_steps}")
    print(f"trainable_params,{sum(parameter.numel() for parameter in trainable_params)}")

    rows: list[dict[str, float | str]] = []
    rows.extend(evaluate_split(model, train_cached, "baseline_train", cfg))
    rows.extend(evaluate_split(model, eval_cached, "baseline_eval", cfg))

    model.eval()
    for step in range(1, args.distill_steps + 1):
        cached_batch = train_cached[(step - 1) % len(train_cached)]
        reset_sva_stats(cfg)
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(**cached_batch.batch, use_cache=False).logits
        loss = distill_kl(
            cached_batch.teacher_logits,
            student_logits,
            cached_batch.batch["attention_mask"],
            args.temperature,
        )
        if args.ce_weight > 0:
            ce_loss = shifted_loss(
                student_logits,
                cached_batch.batch["input_ids"],
                cached_batch.batch["attention_mask"],
            )
            loss = loss + args.ce_weight * ce_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == args.distill_steps):
            print(f"train_step,{step},{cached_batch.mode},{loss.item():.6f}", flush=True)

    rows.extend(evaluate_split(model, train_cached, "distilled_train", cfg))
    rows.extend(evaluate_split(model, eval_cached, "distilled_eval", cfg))
    print_rows(rows)
    print_mean_rows(rows)


if __name__ == "__main__":
    main()
