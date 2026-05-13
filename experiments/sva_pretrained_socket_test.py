"""Socket Summon-Verify Attention into a pretrained Llama-style LM.

The test keeps pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits.
Only the attention lookup changes: full causal attention is replaced by an SVA
candidate mask followed by exact QK scoring over the summoned candidates.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


TEXTS = [
    (
        "Summon-Verify Attention asks a query to summon a small set of candidate memories, "
        "then verifies them exactly. The important question is whether a pretrained model's "
        "own keys and queries already contain enough address structure for that lookup to work."
    ),
    (
        "A tiny language model is useful here because its learned keys and queries are real, "
        "but the model is still inspectable. If the socket test works on a small Llama-style "
        "decoder, the next experiment can move to a larger model without changing the premise."
    ),
    (
        "The river moved quietly through the city while the old library kept its lights on late "
        "into the evening. A courier crossed the bridge, checked the address twice, and found "
        "the right door by matching small details that nobody else seemed to notice."
    ),
    (
        "Sparse attention methods must preserve the important behavior of full attention while "
        "reading far fewer cached tokens. The failure mode is simple: if the retrieval stage misses "
        "one crucial earlier token, the exact verifier can only be precisely wrong."
    ),
    (
        "Modern decoder blocks usually combine grouped-query attention, rotary position embeddings, "
        "RMS normalization, and a gated feedforward network. A socketed replacement should leave those "
        "learned components intact and change only the way candidate keys are selected."
    ),
    (
        "In an engineering notebook, the first result is rarely the final answer. A useful experiment "
        "separates the scientific risk from the systems risk, records what actually happened, and then "
        "makes the next measurement sharper."
    ),
]


@dataclass
class SVAConfig:
    tables: int = 16
    bits: int = 10
    budget: int = 64
    probe_radius: int = 1
    seed: int = 17
    stats: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class SVALlamaAttention(nn.Module):
    def __init__(self, original: nn.Module, cfg: SVAConfig) -> None:
        super().__init__()
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.cfg = cfg

        n_heads = self.config.num_attention_heads
        generator = torch.Generator(device=self.q_proj.weight.device)
        generator.manual_seed(cfg.seed + 10_000 * int(self.layer_idx or 0))
        projections = torch.randn(
            n_heads,
            cfg.tables,
            cfg.bits,
            self.head_dim,
            generator=generator,
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )
        projections = projections / math.sqrt(self.head_dim)
        self.register_buffer("sva_projections", projections, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("SVA socket test requires model-provided RoPE position embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output = self.sva_attention(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output), None

    def sva_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.cfg
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        projections = self.sva_projections[:n_heads, : cfg.tables, : cfg.bits, :].to(dtype=query.dtype)

        q_bits = torch.einsum("bhtd,hrmd->bhtrm", query, projections) > 0
        k_bits = torch.einsum("bhsd,hrmd->bhsrm", key_states, projections) > 0
        hamming = (q_bits[:, :, :, None, :, :] != k_bits[:, :, None, :, :, :]).sum(dim=-1)
        candidate_mask = (hamming <= cfg.probe_radius).any(dim=-1)

        if attention_mask is not None:
            allowed = attention_mask[..., :q_len, :k_len] > -1e4
            candidate_mask = candidate_mask & allowed
        else:
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril()
            candidate_mask = candidate_mask & causal[None, None, :, :]

        if q_len == k_len:
            eye = torch.eye(q_len, dtype=torch.bool, device=query.device)
            candidate_mask = candidate_mask | eye[None, None, :, :]

        scores = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        candidate_counts = candidate_mask.sum(dim=-1)

        if cfg.budget > 0 and cfg.budget < k_len:
            chosen_scores, chosen_idx = torch.topk(scores, cfg.budget, dim=-1)
            source = value_states[:, :, None, :, :].expand(batch, n_heads, q_len, k_len, head_dim)
            chosen_values = torch.gather(
                source,
                dim=3,
                index=chosen_idx[..., None].expand(batch, n_heads, q_len, cfg.budget, head_dim),
            )
            weights = F.softmax(chosen_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output = (weights[..., None] * chosen_values).sum(dim=-2)
            verified_counts = torch.minimum(
                candidate_counts,
                torch.tensor(cfg.budget, dtype=candidate_counts.dtype, device=candidate_counts.device),
            )
        else:
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output = torch.matmul(weights, value_states)
            verified_counts = candidate_counts

        cfg.stats["summoned"] += float(candidate_counts.sum().item())
        cfg.stats["verified"] += float(verified_counts.sum().item())
        cfg.stats["queries"] += float(candidate_counts.numel())
        return output


def patch_model_with_sva(model: nn.Module, cfg: SVAConfig) -> None:
    for layer in model.model.layers:
        layer.self_attn = SVALlamaAttention(layer.self_attn, cfg)


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return TEXTS
    with open(path, "r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts found in {path}")
    return texts


def encode_batch(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in batch.items()}


def shifted_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compare_logits(
    full_logits: torch.Tensor,
    sva_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, float]:
    valid = attention_mask[:, 1:].bool()
    full = full_logits[:, :-1, :][valid]
    sva = sva_logits[:, :-1, :][valid]
    full_log_probs = F.log_softmax(full.float(), dim=-1)
    sva_log_probs = F.log_softmax(sva.float(), dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - sva_log_probs)).sum(dim=-1).mean()
    top1_agreement = (full.argmax(dim=-1) == sva.argmax(dim=-1)).float().mean()
    cosine = F.cosine_similarity(full.float(), sva.float(), dim=-1).mean()
    return {
        "kl_to_full": float(kl.item()),
        "top1_agreement": float(top1_agreement.item()),
        "logit_cosine": float(cosine.item()),
    }


@torch.no_grad()
def run_model(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    output = model(**batch, use_cache=False)
    return output.logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained LLM SVA socket test.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

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

    texts = load_texts(args.text_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    batch = encode_batch(tokenizer, texts, args.max_length, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    full_logits = run_model(model, batch)
    full_loss = shifted_loss(full_logits, batch["input_ids"], batch["attention_mask"])

    cfg = SVAConfig(args.tables, args.bits, args.budget, args.probe_radius, args.seed)
    patch_model_with_sva(model, cfg)
    sva_logits = run_model(model, batch)
    sva_loss = shifted_loss(sva_logits, batch["input_ids"], batch["attention_mask"])
    metrics = compare_logits(full_logits, sva_logits, batch["attention_mask"])

    avg_summoned = cfg.stats["summoned"] / max(cfg.stats["queries"], 1.0)
    avg_verified = cfg.stats["verified"] / max(cfg.stats["queries"], 1.0)
    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"n_texts,{len(texts)}")
    print(f"seq_len,{batch['input_ids'].shape[1]}")
    print(f"tables,{args.tables}")
    print(f"bits,{args.bits}")
    print(f"budget,{args.budget}")
    print(f"probe_radius,{args.probe_radius}")
    print(f"full_loss,{full_loss.item():.6f}")
    print(f"sva_loss,{sva_loss.item():.6f}")
    print(f"loss_delta,{(sva_loss - full_loss).item():.6f}")
    for key, value in metrics.items():
        print(f"{key},{value:.6f}")
    print(f"avg_summoned,{avg_summoned:.3f}")
    print(f"avg_verified,{avg_verified:.3f}")


if __name__ == "__main__":
    main()
