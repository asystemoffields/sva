from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sva import SVAArtifactBundle, SVALayerArtifacts, SVALlamaAttention, patch_llama_attention


def make_tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def make_tiny_bundle() -> SVAArtifactBundle:
    generator = torch.Generator().manual_seed(11)
    layers = {}
    for layer_idx in range(2):
        layers[layer_idx] = SVALayerArtifacts(
            q_proj=torch.randn(2, 8, 4, generator=generator),
            k_proj=torch.randn(2, 8, 4, generator=generator),
            logit_scale=torch.zeros(2),
            coarse_codebooks=torch.randn(2, 2, 8, 2, generator=generator),
            train_loss=0.0,
            hard_loss=0.0,
            route_source="qk",
        )
    return SVAArtifactBundle(
        manifest={
            "schema_version": 1,
            "artifact_type": "summon_verify_attention",
            "rank_dim": 4,
            "coarse_subspaces": 2,
            "coarse_codewords": 8,
            "default_shortlist": 4,
            "default_budget": 2,
            "layer_count": 2,
            "layers": [0, 1],
        },
        layers=layers,
    )


class SVALlamaAdapterTest(unittest.TestCase):
    def test_static_refill_selects_unique_candidates(self) -> None:
        candidate_idx = torch.tensor([[4, 4, 3, 3, 2, 1]])
        candidate_scores = torch.tensor([[9.0, 8.5, 8.0, 7.5, 7.0, 6.0]])
        candidate_valid = torch.ones_like(candidate_idx, dtype=torch.bool)

        selected_idx, selected_scores, selected_valid, refill_count = SVALlamaAttention._top_unique_candidates(
            candidate_idx,
            candidate_scores,
            candidate_valid,
            budget=3,
            refill_factor=2,
        )

        self.assertEqual(refill_count, 6)
        self.assertEqual(selected_idx[selected_valid].tolist(), [4, 3, 2])
        self.assertEqual(selected_scores[selected_valid].tolist(), [9.0, 8.0, 7.0])

    def test_patch_prefill_unpatch(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original = model.model.layers[0].self_attn

        handle = patch_llama_attention(model, bundle)
        self.assertIsInstance(model.model.layers[0].self_attn, SVALlamaAttention)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=False)

        self.assertEqual(tuple(output.logits.shape), (1, 6, 64))
        summary = handle.stats.summary()
        self.assertGreater(summary["queries"], 0)
        self.assertLessEqual(summary["avg_verified"], 2.0)

        handle.unpatch()
        self.assertIs(model.model.layers[0].self_attn, original)

    def test_context_manager_restores_after_cached_decode(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original = model.model.layers[1].self_attn

        with patch_llama_attention(
            model,
            bundle,
            shortlist=4,
            budget=2,
            summon_mode="inverted",
            inverted_cells_per_subspace=2,
            adaptive_min_budget=1,
            adaptive_mid_budget=2,
        ) as handle:
            with torch.no_grad():
                first = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
                second = model(input_ids=torch.tensor([[5]]), use_cache=True, past_key_values=first.past_key_values)
            self.assertEqual(tuple(second.logits.shape), (1, 1, 64))
            self.assertGreater(handle.stats.summary()["queries"], 0)
            self.assertGreater(handle.stats.summary()["avg_cell_visits"], 0)

        self.assertIs(model.model.layers[1].self_attn, original)

    def test_static_inverted_cached_decode(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original = model.model.layers[1].self_attn

        with patch_llama_attention(
            model,
            bundle,
            shortlist=4,
            budget=2,
            summon_mode="inverted_static",
            inverted_cells_per_subspace=2,
            profile_components=True,
        ) as handle:
            with torch.no_grad():
                first = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
                second = model(input_ids=torch.tensor([[5]]), use_cache=True, past_key_values=first.past_key_values)
                third = model(input_ids=torch.tensor([[6]]), use_cache=True, past_key_values=second.past_key_values)
            self.assertEqual(tuple(second.logits.shape), (1, 1, 64))
            self.assertEqual(tuple(third.logits.shape), (1, 1, 64))
            summary = handle.stats.summary()
            self.assertGreater(summary["queries"], 0)
            self.assertGreater(summary["avg_refill_pool"], 0)
            self.assertGreaterEqual(summary["avg_static_total_ms"], 0)
            self.assertGreaterEqual(summary["avg_static_outer_total_ms"], summary["avg_static_total_ms"])
            attention = model.model.layers[1].self_attn
            self.assertIsInstance(attention, SVALlamaAttention)
            self.assertEqual(attention._cached_key_len, 6)
            self.assertEqual(attention._cached_postings_key_len, 4)
            self.assertEqual(tuple(attention._cached_coarse_codes.shape), (2, 4, 2))

        self.assertIs(model.model.layers[1].self_attn, original)

    def test_static_inverted_rebuilds_after_tail_interval(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original = model.model.layers[1].self_attn

        with patch_llama_attention(
            model,
            bundle,
            shortlist=4,
            budget=2,
            summon_mode="inverted_static",
            inverted_cells_per_subspace=2,
            static_tail_rebuild_interval=2,
        ):
            with torch.no_grad():
                first = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
                second = model(input_ids=torch.tensor([[5]]), use_cache=True, past_key_values=first.past_key_values)
                model(input_ids=torch.tensor([[6]]), use_cache=True, past_key_values=second.past_key_values)
            attention = model.model.layers[1].self_attn
            self.assertIsInstance(attention, SVALlamaAttention)
            self.assertEqual(attention._cached_key_len, 6)
            self.assertEqual(attention._cached_postings_key_len, 6)
            self.assertEqual(tuple(attention._cached_coarse_codes.shape), (2, 6, 2))

        self.assertIs(model.model.layers[1].self_attn, original)

    def test_static_inverted_multitoken_cache_keeps_full_codes(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original = model.model.layers[1].self_attn

        with patch_llama_attention(
            model,
            bundle,
            shortlist=4,
            budget=2,
            summon_mode="inverted_static",
            inverted_cells_per_subspace=2,
        ):
            with torch.no_grad():
                first = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
                second = model(input_ids=torch.tensor([[5, 6]]), use_cache=True, past_key_values=first.past_key_values)
            self.assertEqual(tuple(second.logits.shape), (1, 2, 64))
            attention = model.model.layers[1].self_attn
            self.assertIsInstance(attention, SVALlamaAttention)
            self.assertEqual(attention._cached_key_len, 6)
            self.assertEqual(tuple(attention._cached_coarse_codes.shape), (2, 6, 2))

        self.assertIs(model.model.layers[1].self_attn, original)

    def test_patch_selected_layers_only(self) -> None:
        model = make_tiny_model()
        bundle = make_tiny_bundle()
        original_layer0 = model.model.layers[0].self_attn
        original_layer1 = model.model.layers[1].self_attn

        handle = patch_llama_attention(model, bundle, layers=[1])
        self.assertIs(model.model.layers[0].self_attn, original_layer0)
        self.assertIsInstance(model.model.layers[1].self_attn, SVALlamaAttention)

        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
        self.assertEqual(tuple(output.logits.shape), (1, 4, 64))
        self.assertGreater(handle.stats.summary()["queries"], 0)

        handle.unpatch()
        self.assertIs(model.model.layers[0].self_attn, original_layer0)
        self.assertIs(model.model.layers[1].self_attn, original_layer1)


if __name__ == "__main__":
    unittest.main()
