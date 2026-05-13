"""Trainable associative-recall test for Summon-Verify Attention.

This is the first learned-representation checkpoint. A tiny modern causal
decoder is trained with full attention on random key/value recall. After
training, the same weights are evaluated with full causal attention and with SVA
replacing each attention layer.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nearby_codes(code: int, n_bits: int, radius: int) -> list[int]:
    codes = [code]
    if radius <= 0:
        return codes
    bit_masks = [1 << bit for bit in range(n_bits)]
    for distance in range(1, min(radius, n_bits) + 1):
        for masks in combinations(bit_masks, distance):
            flip = 0
            for mask in masks:
                flip ^= mask
            codes.append(code ^ flip)
    return codes


def make_batch(args: argparse.Namespace, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = 2 * args.n_pairs + 2
    x = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.empty((batch_size,), dtype=torch.long, device=device)
    query_marker = args.n_keys + args.n_values

    for batch_idx in range(batch_size):
        keys = torch.randperm(args.n_keys, device=device)[: args.n_pairs]
        values = torch.randint(0, args.n_values, (args.n_pairs,), device=device)
        query_pair = int(torch.randint(0, args.n_pairs, ()).item())
        x[batch_idx, 0 : 2 * args.n_pairs : 2] = keys
        x[batch_idx, 1 : 2 * args.n_pairs : 2] = args.n_keys + values
        x[batch_idx, -2] = query_marker
        x[batch_idx, -1] = keys[query_pair]
        y[batch_idx] = args.n_keys + values[query_pair]

    return x, y


class SVAConfig:
    def __init__(self, tables: int, bits: int, budget: int, probe_radius: int, impl: str) -> None:
        self.tables = tables
        self.bits = bits
        self.budget = budget
        self.probe_radius = probe_radius
        self.impl = impl


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    seq_len = x.shape[-2]
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freqs / head_dim))
    angles = torch.einsum("t,d->td", positions, inv_freq).to(dtype=x.dtype)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_sva_tables: int,
        max_sva_bits: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        projections = torch.randn(n_heads, max_sva_tables, max_sva_bits, self.head_dim)
        projections = projections / math.sqrt(self.head_dim)
        self.register_buffer("sva_projections", projections)

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        sva: SVAConfig | None,
        stats: dict[str, float] | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_base)
        k = apply_rope(k, self.rope_base)

        if mode == "full":
            y = self.full_attention(q, k, v)
        elif mode == "sva":
            if sva is None:
                raise ValueError("SVA config is required for mode='sva'")
            if sva.impl == "loop":
                y = self.sva_attention_loop(q, k, v, sva, stats)
            elif sva.impl == "mask":
                y = self.sva_attention_mask(q, k, v, sva, stats)
            else:
                raise ValueError(f"unknown SVA implementation: {sva.impl}")
        else:
            raise ValueError(f"unknown attention mode: {mode}")

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out(y)

    def full_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v

    def sva_attention_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sva: SVAConfig,
        stats: dict[str, float] | None,
    ) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = q.shape
        projections = self.sva_projections[:, : sva.tables, : sva.bits, :]
        q_bits = torch.einsum("bhtd,hrmd->bhtrm", q, projections) > 0
        k_bits = torch.einsum("bhtd,hrmd->bhtrm", k, projections) > 0

        hamming = (q_bits[:, :, :, None, :, :] != k_bits[:, :, None, :, :, :]).sum(dim=-1)
        candidate_mask = (hamming <= sva.probe_radius).any(dim=-1)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
        candidate_mask = candidate_mask & causal_mask[None, None, :, :]
        eye = torch.eye(seq_len, dtype=torch.bool, device=q.device)
        candidate_mask = candidate_mask | eye[None, None, :, :]

        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        candidate_counts = candidate_mask.sum(dim=-1)

        if sva.budget > 0 and sva.budget < seq_len:
            chosen_scores, chosen_idx = torch.topk(scores, sva.budget, dim=-1)
            source = v[:, :, None, :, :].expand(batch, n_heads, seq_len, seq_len, head_dim)
            chosen_v = torch.gather(
                source,
                dim=3,
                index=chosen_idx[..., None].expand(batch, n_heads, seq_len, sva.budget, head_dim),
            )
            weights = F.softmax(chosen_scores, dim=-1)
            out = (weights[..., None] * chosen_v).sum(dim=-2)
            verified_counts = torch.minimum(
                candidate_counts,
                torch.tensor(sva.budget, dtype=candidate_counts.dtype, device=candidate_counts.device),
            )
        else:
            weights = F.softmax(scores, dim=-1)
            out = weights @ v
            verified_counts = candidate_counts

        if stats is not None:
            stats["summoned"] += float(candidate_counts.sum().item())
            stats["verified"] += float(verified_counts.sum().item())
            stats["queries"] += float(batch * n_heads * seq_len)
        return out

    def sva_attention_loop(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sva: SVAConfig,
        stats: dict[str, float] | None,
    ) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(v)
        projections = self.sva_projections[:, : sva.tables, : sva.bits, :]
        powers = (1 << torch.arange(sva.bits, device=q.device, dtype=torch.long))

        for batch_idx in range(batch):
            for head_idx in range(n_heads):
                key_codes = self.codes(k[batch_idx, head_idx], projections[head_idx], powers)
                query_codes = self.codes(q[batch_idx, head_idx], projections[head_idx], powers)
                for timestep in range(seq_len):
                    candidates: set[int] = set()
                    for table_idx in range(sva.tables):
                        code = int(query_codes[timestep, table_idx].item())
                        near = nearby_codes(code, sva.bits, sva.probe_radius)
                        prefix_codes = key_codes[: timestep + 1, table_idx]
                        for near_code in near:
                            hits = torch.nonzero(prefix_codes == near_code, as_tuple=False).flatten()
                            candidates.update(int(hit.item()) for hit in hits)

                    if not candidates:
                        candidates.add(timestep)

                    candidate_idx = torch.tensor(sorted(candidates), dtype=torch.long, device=q.device)
                    scores = (
                        q[batch_idx, head_idx, timestep] @ k[batch_idx, head_idx, candidate_idx].T
                    ) / math.sqrt(head_dim)
                    if sva.budget > 0 and candidate_idx.numel() > sva.budget:
                        chosen_scores, chosen_order = torch.topk(scores, sva.budget)
                        chosen_idx = candidate_idx[chosen_order]
                    else:
                        chosen_scores = scores
                        chosen_idx = candidate_idx
                    weights = F.softmax(chosen_scores, dim=-1)
                    out[batch_idx, head_idx, timestep] = weights @ v[batch_idx, head_idx, chosen_idx]

                    if stats is not None:
                        stats["summoned"] += float(candidate_idx.numel())
                        stats["verified"] += float(chosen_idx.numel())
                        stats["queries"] += 1.0
        return out

    @staticmethod
    def codes(vectors: torch.Tensor, projections: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
        dots = torch.einsum("td,rbd->trb", vectors, projections)
        bits = dots > 0
        return (bits.long() * powers).sum(dim=-1)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_mult: int,
        max_sva_tables: int,
        max_sva_bits: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_sva_tables, max_sva_bits, rope_base)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult * d_model)

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        sva: SVAConfig | None,
        stats: dict[str, float] | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mode, sva, stats)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyRecallTransformer(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.vocab_size = args.n_keys + args.n_values + 1
        self.seq_len = 2 * args.n_pairs + 2
        self.token_emb = nn.Embedding(self.vocab_size, args.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    args.d_model,
                    args.n_heads,
                    args.mlp_mult,
                    max(args.sva_tables),
                    args.sva_bits,
                    args.rope_base,
                )
                for _ in range(args.n_layers)
            ]
        )
        self.ln = RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, self.vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "full",
        sva: SVAConfig | None = None,
        stats: dict[str, float] | None = None,
    ) -> torch.Tensor:
        h = self.token_emb(x)
        for block in self.blocks:
            h = block(h, mode, sva, stats)
        h = self.ln(h)
        return self.head(h[:, -1])


@torch.no_grad()
def evaluate(
    model: TinyRecallTransformer,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    sva: SVAConfig | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    stats: dict[str, float] = defaultdict(float)
    for _ in range(args.eval_batches):
        x, y = make_batch(args, args.eval_batch_size, device)
        logits = model(x, mode=mode, sva=sva, stats=stats if mode == "sva" else None)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_examples += y.numel()
    result = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "avg_summoned": 0.0,
        "avg_verified": 0.0,
    }
    if stats["queries"] > 0:
        result["avg_summoned"] = stats["summoned"] / stats["queries"]
        result["avg_verified"] = stats["verified"] / stats["queries"]
    return result


def train(args: argparse.Namespace, device: torch.device) -> TinyRecallTransformer:
    start = time.perf_counter()
    model = TinyRecallTransformer(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(args, args.batch_size, device)
        logits = model(x, mode="full")
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            result = evaluate(model, args, device, "full")
            print(
                f"train_step,{step},"
                f"loss,{loss.item():.4f},"
                f"val_loss,{result['loss']:.4f},"
                f"val_acc,{result['accuracy']:.4f}",
                flush=True,
            )
    print(f"train_done_seconds,{time.perf_counter() - start:.2f}", flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full attention, then evaluate SVA with learned Q/K/V.")
    parser.add_argument("--n-keys", type=int, default=96)
    parser.add_argument("--n-values", type=int, default=96)
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sva-tables", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--sva-bits", type=int, default=10)
    parser.add_argument("--sva-budget", type=int, default=16)
    parser.add_argument("--sva-impl", choices=["mask", "loop"], default="mask")
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device,{device}", flush=True)
    model = train(args, device)

    print("method,loss,accuracy,avg_summoned,avg_verified")
    eval_start = time.perf_counter()
    print("eval_start,full_attention", flush=True)
    full = evaluate(model, args, device, "full")
    print(f"full_attention,{full['loss']:.4f},{full['accuracy']:.4f},0.0,0.0")
    print(f"eval_done,full_attention,{time.perf_counter() - eval_start:.2f}", flush=True)
    for tables in args.sva_tables:
        method = f"sva_{tables}x{args.sva_bits}"
        eval_start = time.perf_counter()
        print(f"eval_start,{method}", flush=True)
        exact = evaluate(
            model,
            args,
            device,
            "sva",
            SVAConfig(tables, args.sva_bits, args.sva_budget, 0, args.sva_impl),
        )
        print(
            f"{method},"
            f"{exact['loss']:.4f},"
            f"{exact['accuracy']:.4f},"
            f"{exact['avg_summoned']:.1f},"
            f"{exact['avg_verified']:.1f}"
        )
        print(f"eval_done,{method},{time.perf_counter() - eval_start:.2f}", flush=True)
        method = f"sva_probe{args.probe_radius}_{tables}x{args.sva_bits}"
        eval_start = time.perf_counter()
        print(f"eval_start,{method}", flush=True)
        probed = evaluate(
            model,
            args,
            device,
            "sva",
            SVAConfig(tables, args.sva_bits, args.sva_budget, args.probe_radius, args.sva_impl),
        )
        print(
            f"{method},"
            f"{probed['loss']:.4f},"
            f"{probed['accuracy']:.4f},"
            f"{probed['avg_summoned']:.1f},"
            f"{probed['avg_verified']:.1f}"
        )
        print(f"eval_done,{method},{time.perf_counter() - eval_start:.2f}", flush=True)


if __name__ == "__main__":
    main()
