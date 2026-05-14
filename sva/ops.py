"""Core tensor operations for SVA serving."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def assign_to_centroids(x: torch.Tensor, centroids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Assign rows in `x` to nearest centroids using squared Euclidean distance."""

    labels = []
    centroids_f = centroids.float()
    centroid_norms = (centroids_f * centroids_f).sum(dim=-1)
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size].float()
        distances = (chunk * chunk).sum(dim=-1, keepdim=True) - 2.0 * chunk @ centroids_f.T + centroid_norms[None, :]
        labels.append(distances.argmin(dim=-1))
    return torch.cat(labels, dim=0)


@torch.no_grad()
def encode_product_keys(k_low: torch.Tensor, codebooks: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Encode low-rank keys into product-quantized code ids."""

    n_heads, _, rank_dim = k_low.shape
    subspaces = codebooks.shape[1]
    if rank_dim % subspaces != 0:
        raise ValueError(f"rank_dim={rank_dim} must be divisible by subspaces={subspaces}.")

    sub_dim = rank_dim // subspaces
    codes: list[torch.Tensor] = []
    for head_idx in range(n_heads):
        head_codes: list[torch.Tensor] = []
        for subspace_idx in range(subspaces):
            start = subspace_idx * sub_dim
            end = start + sub_dim
            labels = assign_to_centroids(
                k_low[head_idx, :, start:end],
                codebooks[head_idx, subspace_idx],
                chunk_size,
            )
            head_codes.append(labels)
        codes.append(torch.stack(head_codes, dim=-1))
    return torch.stack(codes, dim=0)


@torch.no_grad()
def product_quantized_scores(
    q_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
) -> torch.Tensor:
    """Approximate low-rank QK scores from product-quantized key codes."""

    n_heads, n_queries, _ = q_low.shape
    seq_len = codes.shape[1]
    subspaces = codebooks.shape[1]
    sub_dim = rank_dim // subspaces
    q_parts = q_low.float().reshape(n_heads, n_queries, subspaces, sub_dim)
    scores = torch.zeros(n_heads, n_queries, seq_len, device=q_low.device, dtype=torch.float32)

    for head_idx in range(n_heads):
        for subspace_idx in range(subspaces):
            table = q_parts[head_idx, :, subspace_idx] @ codebooks[head_idx, subspace_idx].float().T
            scores[head_idx] += table[:, codes[head_idx, :, subspace_idx].long()]

    return scores / math.sqrt(rank_dim)
