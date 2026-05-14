"""Logit-distill tiny adapters for a tight-budget late4 SVA socket."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import SVALlamaPatcher, patch_llama_attention
from sva_pretrained_socket_test import format_layer_list, parse_layer_list
from sva_passkey_language_benchmark import PromptCase, build_prompt_case, comma_ints, comma_strings


@dataclass
class CachedPrompt:
    key: str
    placement: str
    context: int
    target: str
    case: PromptCase
    teacher_logits: torch.Tensor


class SVAResidualOutputAdapter(nn.Module):
    def __init__(self, sva_attn: nn.Module, hidden_size: int, rank: int, scale: float) -> None:
        super().__init__()
        self.sva_attn = sva_attn
        self.scale = scale
        self.down = nn.Linear(hidden_size * 2, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        attn_output, attn_weights = self.sva_attn(hidden_states, *args, **kwargs)
        adapter_input = torch.cat([hidden_states.float(), attn_output.float()], dim=-1)
        correction = self.up(F.gelu(self.down(adapter_input))) * self.scale
        return attn_output + correction.to(dtype=attn_output.dtype), attn_weights


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace("\n", " ").replace(",", ";")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def final_logit_metrics(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> dict[str, float]:
    teacher = teacher_logits.float()
    student = student_logits.float()
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    top1 = (teacher.argmax(dim=-1) == student.argmax(dim=-1)).float().mean()
    cosine = F.cosine_similarity(teacher, student, dim=-1).mean()
    return {
        "final_kl_to_full": float(kl.item()),
        "final_top1_agreement": float(top1.item()),
        "final_logit_cosine": float(cosine.item()),
    }


def distill_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher = teacher_logits.float() / temperature
    student = student_logits.float() / temperature
    teacher_probs = F.softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def run_logits_for_target(model: nn.Module, case: PromptCase, target: str) -> torch.Tensor:
    if target == "final":
        output = model(
            input_ids=case.input_ids,
            attention_mask=case.attention_mask,
            use_cache=False,
        )
        logits = output.logits[:, -1, :]
        del output
        return logits
    if target == "answer":
        if int(case.answer_ids.numel()) > 1:
            answer_prefix = case.answer_ids[:-1].view(1, -1)
            input_ids = torch.cat([case.input_ids, answer_prefix], dim=1)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = case.input_ids
            attention_mask = case.attention_mask
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        positions = torch.arange(
            case.context - 1,
            case.context - 1 + int(case.answer_ids.numel()),
            device=case.input_ids.device,
        )
        logits = output.logits[0, positions, :]
        del output
        return logits
    raise ValueError(f"Unknown distillation target: {target}")


def cache_teacher_logits(
    model: nn.Module,
    tokenizer,
    keys: list[str],
    placements: list[str],
    contexts: list[int],
    device: torch.device,
    target: str,
) -> list[CachedPrompt]:
    rows: list[CachedPrompt] = []
    model.eval()
    with torch.no_grad():
        for context in contexts:
            for key in keys:
                for placement in placements:
                    case = build_prompt_case(tokenizer, context, key, placement, device)
                    teacher_logits = run_logits_for_target(model, case, target)
                    rows.append(
                        CachedPrompt(
                            key=key,
                            placement=placement,
                            context=context,
                            target=target,
                            case=case,
                            teacher_logits=teacher_logits.detach().cpu(),
                        )
                    )
                    del teacher_logits
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    return rows


def wrap_socket_layers_with_adapters(
    model: nn.Module,
    socket_layers: list[int],
    rank: int,
    scale: float,
) -> None:
    hidden_size = int(model.config.hidden_size)
    for layer_idx in socket_layers:
        layer = model.model.layers[layer_idx]
        device = next(layer.self_attn.parameters()).device
        layer.self_attn = SVAResidualOutputAdapter(layer.self_attn, hidden_size, rank, scale).to(device)


def collect_adapter_state(model: nn.Module, socket_layers: list[int]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer_idx in socket_layers:
        attention = model.model.layers[layer_idx].self_attn
        if not isinstance(attention, SVAResidualOutputAdapter):
            raise TypeError(f"Layer {layer_idx} is not wrapped with SVAResidualOutputAdapter.")
        state[f"layers.{layer_idx}.down.weight"] = attention.down.weight.detach().cpu()
        state[f"layers.{layer_idx}.up.weight"] = attention.up.weight.detach().cpu()
    return state


def save_adapter_bundle(
    output_dir: Path,
    model: nn.Module,
    socket_layers: list[int],
    args: argparse.Namespace,
    results: list[dict[str, float | int | str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "sva_residual_output_adapter_v1",
        "model_id": args.model_id,
        "artifact_dir": str(args.artifact_dir),
        "socket_layers": socket_layers,
        "shortlist": args.shortlist,
        "budget": args.budget,
        "query_chunk_size": args.query_chunk_size,
        "summon_mode": args.summon_mode,
        "adapter_rank": args.adapter_rank,
        "adapter_scale": args.adapter_scale,
        "distill_steps": args.distill_steps,
        "lr": args.lr,
        "temperature": args.temperature,
        "target": args.target,
        "train_contexts": args.contexts,
        "train_keys": args.train_keys,
        "eval_keys": args.eval_keys,
        "train_placements": args.train_placements,
        "eval_placements": args.eval_placements,
        "results": results,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save({"state_dict": collect_adapter_state(model, socket_layers)}, output_dir / "adapter_weights.pt")


def load_adapter_bundle(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    manifest = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    payload = torch.load(adapter_dir / "adapter_weights.pt", map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected adapter weights payload in {adapter_dir}.")
    return manifest, state_dict


def load_adapter_weights(model: nn.Module, socket_layers: list[int], state_dict: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for layer_idx in socket_layers:
            attention = model.model.layers[layer_idx].self_attn
            if not isinstance(attention, SVAResidualOutputAdapter):
                raise TypeError(f"Layer {layer_idx} is not wrapped with SVAResidualOutputAdapter.")
            attention.down.weight.copy_(state_dict[f"layers.{layer_idx}.down.weight"].to(attention.down.weight.device))
            attention.up.weight.copy_(state_dict[f"layers.{layer_idx}.up.weight"].to(attention.up.weight.device))


def run_student_final_logits(
    model: nn.Module,
    patcher: SVALlamaPatcher,
    cached: CachedPrompt,
) -> torch.Tensor:
    patcher.reset_catalogs()
    return run_logits_for_target(model, cached.case, cached.target)


def evaluate(
    model: nn.Module,
    patcher: SVALlamaPatcher,
    rows: list[CachedPrompt],
    phase: str,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    results: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for cached in rows:
            student_logits = run_student_final_logits(model, patcher, cached)
            metrics = final_logit_metrics(cached.teacher_logits.to(device), student_logits)
            row = {
                "phase": phase,
                "context": cached.context,
                "key": cached.key,
                "placement": cached.placement,
                **metrics,
            }
            emit("distill_eval", row)
            results.append(row)
            del student_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def emit_means(rows: list[dict[str, float | int | str]]) -> None:
    by_phase: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_phase.setdefault(str(row["phase"]), []).append(row)
    for phase, phase_rows in by_phase.items():
        emit(
            "distill_mean",
            {
                "phase": phase,
                "final_kl_to_full": sum(float(row["final_kl_to_full"]) for row in phase_rows) / len(phase_rows),
                "final_top1_agreement": sum(float(row["final_top1_agreement"]) for row in phase_rows) / len(phase_rows),
                "final_logit_cosine": sum(float(row["final_logit_cosine"]) for row in phase_rows) / len(phase_rows),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill tight-budget late4 SVA adapters against full-attention logits.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1"))
    parser.add_argument("--contexts", default="32768")
    parser.add_argument("--train-keys", default="731942,184029")
    parser.add_argument("--eval-keys", default="905317")
    parser.add_argument("--train-placements", default="start,middle")
    parser.add_argument("--eval-placements", default="start,middle,end")
    parser.add_argument("--socket-layers", default="26-29")
    parser.add_argument("--shortlist", type=int, default=512)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--summon-mode", choices=["scan", "inverted"], default="scan")
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--distill-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--target", choices=["final", "answer"], default="final")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype_map = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    contexts = comma_ints(args.contexts)
    train_keys = comma_strings(args.train_keys)
    eval_keys = comma_strings(args.eval_keys)
    train_placements = comma_strings(args.train_placements)
    eval_placements = comma_strings(args.eval_placements)
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    if socket_layers is None:
        raise ValueError("This distillation probe expects explicit socket layers.")

    print("late4_logit_distill_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"artifact_dir,{args.artifact_dir}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"train_keys,{','.join(train_keys)}", flush=True)
    print(f"eval_keys,{','.join(eval_keys)}", flush=True)
    print(f"train_placements,{','.join(train_placements)}", flush=True)
    print(f"eval_placements,{','.join(eval_placements)}", flush=True)
    print(f"socket_layers,{format_layer_list(socket_layers)}", flush=True)
    print(f"shortlist,{args.shortlist}", flush=True)
    print(f"budget,{args.budget}", flush=True)
    print(f"adapter_rank,{args.adapter_rank}", flush=True)
    print(f"distill_steps,{args.distill_steps}", flush=True)
    print(f"target,{args.target}", flush=True)

    train_rows = cache_teacher_logits(model, tokenizer, train_keys, train_placements, contexts, device, args.target)
    eval_rows = cache_teacher_logits(model, tokenizer, eval_keys, eval_placements, contexts, device, args.target)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patcher = patch_llama_attention(
        model,
        args.artifact_dir,
        shortlist=args.shortlist,
        budget=args.budget,
        query_chunk_size=args.query_chunk_size,
        summon_mode=args.summon_mode,
        layers=socket_layers,
    )
    wrap_socket_layers_with_adapters(model, socket_layers, args.adapter_rank, args.adapter_scale)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"trainable_params,{sum(parameter.numel() for parameter in trainable)}", flush=True)

    all_results: list[dict[str, float | int | str]] = []
    all_results.extend(evaluate(model, patcher, train_rows, "baseline_train", device))
    all_results.extend(evaluate(model, patcher, eval_rows, "baseline_eval", device))

    for step in range(1, args.distill_steps + 1):
        cached = train_rows[(step - 1) % len(train_rows)]
        optimizer.zero_grad(set_to_none=True)
        student_logits = run_student_final_logits(model, patcher, cached)
        teacher_logits = cached.teacher_logits.to(device)
        loss = distill_kl(teacher_logits, student_logits, args.temperature)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == args.distill_steps):
            emit(
                "distill_train",
                {
                    "step": step,
                    "context": cached.context,
                    "key": cached.key,
                    "placement": cached.placement,
                    "loss": float(loss.item()),
                },
            )
        del student_logits, teacher_logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_results.extend(evaluate(model, patcher, train_rows, "distilled_train", device))
    all_results.extend(evaluate(model, patcher, eval_rows, "distilled_eval", device))
    emit_means(all_results)
    if args.output_dir is not None:
        save_adapter_bundle(args.output_dir, model, socket_layers, args, all_results)
        print(f"adapter_output_dir,{args.output_dir}", flush=True)
    print("late4_logit_distill_done", flush=True)


if __name__ == "__main__":
    main()
