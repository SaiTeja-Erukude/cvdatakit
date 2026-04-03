"""Tests for cvdatakit.active_learning."""

from __future__ import annotations

import numpy as np
import pytest

from cvdatakit.active_learning.strategies.diversity import DiversityStrategy
from cvdatakit.active_learning.strategies.error_localization import ErrorLocalizationStrategy
from cvdatakit.active_learning.strategies.uncertainty import UncertaintyStrategy


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pool_probs():
    rng = np.random.default_rng(0)
    return rng.dirichlet(np.ones(6), size=200).astype(np.float32)


@pytest.fixture
def pool_embeddings():
    rng = np.random.default_rng(1)
    return rng.standard_normal((200, 64)).astype(np.float32)


# ── UncertaintyStrategy ───────────────────────────────────────────────────────

class TestUncertaintyStrategy:
    @pytest.mark.parametrize("method", ["entropy", "margin", "least_confidence", "bald"])
    def test_score_shape(self, pool_probs, method):
        strat = UncertaintyStrategy(method)
        scores = strat.score(pool_probs)
        assert scores.shape == (200,)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            UncertaintyStrategy("bad_method")

    def test_query_returns_correct_count(self, pool_probs):
        strat = UncertaintyStrategy("entropy")
        budget = 30
        indices = strat.query(pool_probs, budget)
        assert len(indices) == budget

    def test_query_indices_unique(self, pool_probs):
        strat = UncertaintyStrategy("margin")
        indices = strat.query(pool_probs, 50)
        assert len(set(indices.tolist())) == 50

    def test_query_with_index_remapping(self, pool_probs):
        rng = np.random.default_rng(5)
        unlabeled_indices = rng.choice(1000, size=200, replace=False)
        strat = UncertaintyStrategy("entropy")
        indices = strat.query(pool_probs, 20, unlabeled_indices=unlabeled_indices)
        # All returned indices should be in the original index space
        assert all(idx in set(unlabeled_indices.tolist()) for idx in indices.tolist())

    def test_bald_with_mc_samples(self, pool_probs):
        rng = np.random.default_rng(2)
        mc = rng.dirichlet(np.ones(6), size=(10, 200)).astype(np.float32)
        strat = UncertaintyStrategy("bald")
        scores = strat.score(pool_probs, mc_samples=mc)
        assert scores.shape == (200,)
        assert np.all(scores >= 0)

    def test_ranked_has_required_keys(self, pool_probs):
        strat = UncertaintyStrategy("least_confidence")
        ranked = strat.ranked(pool_probs)
        assert len(ranked) == 200
        for item in ranked[:5]:
            assert "index" in item
            assert "uncertainty_score" in item
            assert "predicted_label" in item
            assert "max_prob" in item

    def test_ranked_is_descending(self, pool_probs):
        strat = UncertaintyStrategy("entropy")
        ranked = strat.ranked(pool_probs)
        scores = [r["uncertainty_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_uniform_distribution_highest_entropy(self):
        """Uniform distributions should have the highest entropy scores."""
        k = 5
        uniform = np.full((1, k), 1.0 / k, dtype=np.float32)
        peaked = np.zeros((1, k), dtype=np.float32)
        peaked[0, 0] = 1.0

        strat = UncertaintyStrategy("entropy")
        s_uniform = strat.score(uniform)[0]
        s_peaked = strat.score(peaked)[0]
        assert s_uniform > s_peaked


# ── DiversityStrategy ─────────────────────────────────────────────────────────

class TestDiversityStrategy:
    @pytest.mark.parametrize("method", ["coreset", "cluster_margin", "minmax"])
    def test_query_returns_correct_count(self, pool_embeddings, pool_probs, method):
        strat = DiversityStrategy(method)
        budget = 30
        indices = strat.query(
            pool_embeddings,
            budget,
            pred_probs=pool_probs,  # needed for cluster_margin
        )
        assert len(indices) == budget

    def test_query_indices_in_range(self, pool_embeddings):
        strat = DiversityStrategy("coreset")
        indices = strat.query(pool_embeddings, 40)
        assert all(0 <= i < 200 for i in indices.tolist())

    def test_coreset_with_labeled(self, pool_embeddings):
        rng = np.random.default_rng(0)
        labeled_emb = rng.standard_normal((50, 64)).astype(np.float32)
        strat = DiversityStrategy("coreset")
        indices = strat.query(pool_embeddings, 20, labeled_embeddings=labeled_emb)
        assert len(indices) == 20

    def test_budget_larger_than_pool_capped(self, pool_embeddings, pool_probs):
        strat = DiversityStrategy("coreset")
        indices = strat.query(pool_embeddings, budget=500)
        # Should return at most len(pool_embeddings) samples
        assert len(indices) <= 200

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            DiversityStrategy("invalid")

    def test_cluster_margin_requires_probs(self, pool_embeddings):
        strat = DiversityStrategy("cluster_margin")
        with pytest.raises(ValueError):
            strat.query(pool_embeddings, 10)

    def test_index_remapping(self, pool_embeddings):
        rng = np.random.default_rng(3)
        unlabeled_indices = rng.choice(5000, size=200, replace=False)
        strat = DiversityStrategy("coreset")
        indices = strat.query(pool_embeddings, 20, unlabeled_indices=unlabeled_indices)
        assert all(idx in set(unlabeled_indices.tolist()) for idx in indices.tolist())


# ── ErrorLocalizationStrategy ─────────────────────────────────────────────────

class TestErrorLocalizationStrategy:
    def test_gradient_norm_score(self):
        rng = np.random.default_rng(0)
        grads = rng.standard_normal((100, 512)).astype(np.float32)
        strat = ErrorLocalizationStrategy("gradient_norm")
        scores = strat.score(gradients=grads)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)

    def test_spatial_entropy_score(self):
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((20, 8, 8, 4)).astype(np.float32)
        strat = ErrorLocalizationStrategy("spatial_entropy")
        scores = strat.score(spatial_logits=logits)
        assert scores.shape == (20,)
        assert np.all(scores >= 0)

    def test_influence_approx_score(self):
        rng = np.random.default_rng(2)
        train_losses = rng.random(80).astype(np.float32)
        val_losses = rng.random(20).astype(np.float32)
        strat = ErrorLocalizationStrategy("influence_approx")
        scores = strat.score(train_losses=train_losses, val_losses=val_losses)
        assert scores.shape == (80,)

    def test_query_returns_correct_count(self):
        rng = np.random.default_rng(0)
        grads = rng.standard_normal((100, 256)).astype(np.float32)
        strat = ErrorLocalizationStrategy("gradient_norm")
        indices = strat.query(20, gradients=grads)
        assert len(indices) == 20

    def test_ranked_sorted_descending(self):
        rng = np.random.default_rng(0)
        grads = rng.standard_normal((50, 128)).astype(np.float32)
        strat = ErrorLocalizationStrategy("gradient_norm")
        ranked = strat.ranked(gradients=grads)
        scores = [r["priority_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_missing_required_arg_raises(self):
        strat = ErrorLocalizationStrategy("gradient_norm")
        with pytest.raises(ValueError):
            strat.score()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ErrorLocalizationStrategy("bad")
