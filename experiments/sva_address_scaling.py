"""Estimate SVA address selectivity as context length grows."""

from __future__ import annotations

import argparse
import math


def hamming_ball(bits: int, radius: int) -> int:
    return sum(math.comb(bits, distance) for distance in range(radius + 1))


def candidate_probability(bits: int, radius: int, tables: int) -> float:
    per_table = hamming_ball(bits, radius) / (2**bits)
    return 1.0 - (1.0 - per_table) ** tables


def main() -> None:
    parser = argparse.ArgumentParser(description="SVA random-address scaling calculator.")
    parser.add_argument("--context", type=int, default=1_000_000)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--min-bits", type=int, default=10)
    parser.add_argument("--max-bits", type=int, default=28)
    args = parser.parse_args()

    average_prefix = args.context / 2.0
    print("bits,hamming_ball,candidate_probability,expected_candidates")
    for bits in range(args.min_bits, args.max_bits + 1):
        ball = hamming_ball(bits, args.radius)
        probability = candidate_probability(bits, args.radius, args.tables)
        expected = average_prefix * probability
        print(f"{bits},{ball},{probability:.8f},{expected:.1f}")


if __name__ == "__main__":
    main()

