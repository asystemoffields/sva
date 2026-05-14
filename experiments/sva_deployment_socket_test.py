"""Deployment-proxy socket test for pretrained SVA artifacts.

This differs from the transductive socket harness: artifacts are built from a
calibration text, then frozen while the socketed model is evaluated on a
separate text.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_pretrained_socket_test import (
    TEXTS,
    build_artifacts_for_hidden_states,
    build_progressive_three_stage_artifacts,
    compare_logits,
    config_from_args,
    encode_batch,
    format_layer_list,
    parse_layer_list,
    patch_model_with_sva,
    print_diagnostics,
    run_model,
    shifted_loss,
)


def make_text(mode: str, repeats: int) -> str:
    if mode == "base":
        paragraphs = TEXTS
    elif mode == "rotate":
        paragraphs = TEXTS[1:] + TEXTS[:1]
    elif mode == "reverse":
        paragraphs = list(reversed(TEXTS))
    elif mode == "odds_evens":
        paragraphs = TEXTS[1::2] + TEXTS[::2]
    else:
        raise ValueError(f"Unknown text mode: {mode}")
    return " ".join(paragraphs * max(repeats, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment-proxy pretrained LLM SVA socket test.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--train-text-mode", choices=["base", "rotate", "reverse", "odds_evens"], default="base")
    parser.add_argument("--eval-text-mode", choices=["base", "rotate", "reverse", "odds_evens"], default="rotate")
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--socket-layers",
        default="",
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
    parser.add_argument("--diagnose-topk", type=int, default=16)
    parser.add_argument("--head-report-limit", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    args.mode = "three_stage"

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
    train_text = make_text(args.train_text_mode, args.text_repeats)
    eval_text = make_text(args.eval_text_mode, args.text_repeats)
    train_batch = encode_batch(tokenizer, [train_text], args.max_length, device)
    eval_batch = encode_batch(tokenizer, [eval_text], args.max_length, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    with torch.no_grad():
        eval_full_output = model(**eval_batch, use_cache=False)
    full_logits = eval_full_output.logits
    full_loss = shifted_loss(full_logits, eval_batch["input_ids"], eval_batch["attention_mask"])

    if args.artifact_training == "progressive":
        artifacts = build_progressive_three_stage_artifacts(model, train_batch, socket_layers, args, device)
    else:
        with torch.no_grad():
            train_full_output = model(**train_batch, use_cache=False, output_hidden_states=True)
        if train_full_output.hidden_states is None:
            raise ValueError("Artifact training requires hidden states.")
        artifacts = build_artifacts_for_hidden_states(model, train_full_output.hidden_states, socket_layers, args, device)
        del train_full_output

    cfg = config_from_args(args, artifacts)
    patch_model_with_sva(model, cfg, socket_layers)
    sva_logits = run_model(model, eval_batch)
    sva_loss = shifted_loss(sva_logits, eval_batch["input_ids"], eval_batch["attention_mask"])
    metrics = compare_logits(full_logits, sva_logits, eval_batch["attention_mask"])

    avg_summoned = cfg.stats["summoned"] / max(cfg.stats["queries"], 1.0)
    avg_exact = cfg.stats["exact_scored"] / max(cfg.stats["queries"], 1.0)
    avg_verified = cfg.stats["verified"] / max(cfg.stats["queries"], 1.0)
    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"train_text_mode,{args.train_text_mode}")
    print(f"eval_text_mode,{args.eval_text_mode}")
    print(f"text_repeats,{args.text_repeats}")
    print(f"train_seq_len,{train_batch['input_ids'].shape[1]}")
    print(f"eval_seq_len,{eval_batch['input_ids'].shape[1]}")
    print("mode,deployment_proxy")
    print(f"socket_layers,{format_layer_list(socket_layers)}")
    print(f"socket_layer_count,{len(socket_layers) if socket_layers is not None else len(model.model.layers)}")
    print(f"socket_layers_text,{format_layer_list(socket_layers).replace(',', ';')}")
    print(f"route_source,{args.route_source}")
    print(f"artifact_training,{args.artifact_training}")
    print(f"rank_dim,{args.rank_dim}")
    print(f"coarse_subspaces,{args.coarse_subspaces}")
    print(f"coarse_codewords,{args.coarse_codewords}")
    print(f"coarse_shortlist,{args.coarse_shortlist}")
    print(f"budget,{args.budget}")
    print(f"coarse_label_topk,{args.coarse_label_topk}")
    print(f"ranker_train_steps,{args.ranker_train_steps}")
    print(f"coarse_hard_steps,{args.coarse_hard_steps}")
    print(f"coarse_hard_pool,{args.coarse_hard_pool}")
    print(f"weighted_boost,{args.weighted_boost:g}")
    print(f"full_loss,{full_loss.item():.6f}")
    print(f"sva_loss,{sva_loss.item():.6f}")
    print(f"loss_delta,{(sva_loss - full_loss).item():.6f}")
    for key, value in metrics.items():
        print(f"{key},{value:.6f}")
    print(f"avg_summoned,{avg_summoned:.3f}")
    print(f"avg_verified,{avg_verified:.3f}")
    print(f"avg_exact_scored,{avg_exact:.3f}")
    print(f"avg_postscore_attended,{avg_verified:.3f}")
    print_diagnostics(cfg)


if __name__ == "__main__":
    main()
