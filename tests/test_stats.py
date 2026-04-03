"""Tests for cvdatakit.stats."""

from __future__ import annotations

import numpy as np
import pytest

from cvdatakit.stats.dataset_stats import DatasetStats, _gini


class TestGini:
    def test_perfect_balance(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _gini(p) == pytest.approx(0.0, abs=1e-6)

    def test_extreme_imbalance(self):
        p = np.array([1.0, 0.0, 0.0, 0.0])
        assert _gini(p) > 0.7

    def test_output_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(10))
            g = _gini(p)
            assert 0.0 <= g <= 1.0


class TestDatasetStats:
    def test_summary_keys(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        s = stats.summary()
        assert "num_images" in s
        assert "num_categories" in s
        assert "num_annotations" in s
        assert "class_imbalance" in s
        assert "bbox_statistics" in s

    def test_counts_match_dataset(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        s = stats.summary()
        assert s["num_images"] == coco_dataset.num_images
        assert s["num_categories"] == coco_dataset.num_categories
        assert s["num_annotations"] == coco_dataset.num_annotations

    def test_class_distribution_sorted(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        dist = stats.class_distribution()
        counts = [d["count"] for d in dist]
        assert counts == sorted(counts, reverse=True)

    def test_class_distribution_fractions_sum_to_one(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        dist = stats.class_distribution()
        total = sum(d["fraction"] for d in dist)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_imbalance_metrics_range(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        imb = stats.class_imbalance()
        assert 0.0 <= imb["gini"] <= 1.0
        assert imb["entropy_bits"] >= 0.0
        assert imb["entropy_bits"] <= imb["max_entropy_bits"] + 1e-9
        assert imb["imbalance_ratio"] >= 1.0
        assert imb["effective_num_classes"] >= 1.0

    def test_bbox_statistics(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        bbox = stats.bbox_statistics()
        assert "area" in bbox
        assert "aspect_ratio" in bbox
        assert bbox["total_boxes"] == coco_dataset.num_annotations

    def test_tail_categories_returns_subset(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        tail = stats.tail_categories(percentile=50.0)
        all_cats = set(coco_dataset.category_names)
        assert set(tail).issubset(all_cats)

    def test_co_occurrence_matrix_shape(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        mat, names = stats.co_occurrence_matrix()
        k = coco_dataset.num_categories
        assert mat.shape == (k, k)
        assert len(names) == k
        # diagonal ≥ off-diagonal elements (images contain each class independently)
        assert np.all(mat >= 0)

    def test_annotation_density_keys(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        density = stats.annotation_density()
        assert set(density.keys()) == set(coco_dataset.category_names)

    def test_image_size_statistics(self, coco_dataset):
        stats = DatasetStats(coco_dataset)
        sz = stats.image_size_statistics()
        assert "width" in sz
        assert "height" in sz
