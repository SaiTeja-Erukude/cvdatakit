"""Tests for cvdatakit.quality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvdatakit.io.coco_reader import COCODataset
from cvdatakit.quality.annotation_checks import AnnotationChecker, AnnotationIssue, _bbox_iou
from cvdatakit.quality.label_quality import LabelQualityScorer
from cvdatakit.quality.mislabel_detection import MislabelDetector


# ── AnnotationChecker ─────────────────────────────────────────────────────────

class TestAnnotationChecker:
    def test_no_issues_clean_dataset(self, coco_dataset):
        checker = AnnotationChecker(coco_dataset)
        issues = checker.run()
        # Clean synthetic data should have very few or no errors
        error_issues = [i for i in issues if i.severity == "error"]
        assert len(error_issues) == 0

    def test_detects_out_of_bounds(self, tmp_path):
        data = {
            "images": [{"id": 0, "file_name": "a.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "cat", "supercategory": ""}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [90, 90, 50, 50]}
            ],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data))
        ds = COCODataset(p)
        checker = AnnotationChecker(ds)
        issues = checker.run()
        types = [i.issue_type for i in issues]
        assert "out_of_bounds" in types

    def test_detects_tiny_box(self, tmp_path):
        data = {
            "images": [{"id": 0, "file_name": "a.jpg", "width": 500, "height": 500}],
            "categories": [{"id": 0, "name": "cat", "supercategory": ""}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 1, 1]}
            ],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data))
        ds = COCODataset(p)
        checker = AnnotationChecker(ds, min_bbox_area=4.0)
        issues = checker.run()
        types = [i.issue_type for i in issues]
        assert "tiny_box" in types

    def test_detects_near_duplicate(self, tmp_path):
        data = {
            "images": [{"id": 0, "file_name": "a.jpg", "width": 500, "height": 500}],
            "categories": [{"id": 0, "name": "cat", "supercategory": ""}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 100, 100]},
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [11, 11, 99, 99]},
            ],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data))
        ds = COCODataset(p)
        checker = AnnotationChecker(ds, max_overlap_iou=0.5)
        issues = checker.run()
        types = [i.issue_type for i in issues]
        assert "near_duplicate" in types

    def test_summary_keys(self, coco_dataset):
        checker = AnnotationChecker(coco_dataset)
        s = checker.summary()
        assert "total_issues" in s
        assert "by_type" in s
        assert "by_severity" in s
        assert "issues" in s


class TestBboxIou:
    def test_identical_boxes(self):
        assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _bbox_iou([0, 0, 5, 5], [10, 10, 5, 5]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _bbox_iou([0, 0, 10, 10], [5, 5, 10, 10])
        assert 0.0 < iou < 1.0

    def test_empty_box(self):
        assert _bbox_iou([], []) == 0.0


# ── LabelQualityScorer ────────────────────────────────────────────────────────

class TestLabelQualityScorer:
    def test_score_shape(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        for method in ("normalized_margin", "self_confidence", "entropy_weighted"):
            scores = lq.quality_scores(method)
            assert scores.shape == (len(labels),)

    def test_score_range(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        scores = lq.quality_scores("self_confidence")
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_perfect_labels_high_score(self):
        """When probs perfectly match labels, quality should be 1."""
        k = 4
        n = 40
        probs = np.eye(k)[np.arange(n) % k].astype(np.float32)
        labels = np.arange(n) % k
        lq = LabelQualityScorer(probs, labels)
        scores = lq.quality_scores("self_confidence")
        assert np.all(scores == pytest.approx(1.0))

    def test_find_label_issues_returns_mask(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        mask = lq.find_label_issues()
        assert mask.dtype == bool
        assert mask.shape == (len(labels),)

    def test_ranked_issues_sorted(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        issues = lq.ranked_issues(top_k=10)
        assert len(issues) == 10
        scores = [i["quality_score"] for i in issues]
        assert scores == sorted(scores)

    def test_confusion_matrix_shape(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        joint, marginal = lq.confusion_matrix()
        k = probs.shape[1]
        assert joint.shape == (k, k)
        assert marginal.shape == (k,)

    def test_summary_keys(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        s = lq.summary()
        assert "num_samples" in s
        assert "estimated_error_rate" in s
        assert "flagged_count" in s

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            LabelQualityScorer(np.ones((5,)), np.zeros(5, dtype=int))

    def test_invalid_method_raises(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        lq = LabelQualityScorer(probs, labels)
        with pytest.raises(ValueError):
            lq.quality_scores("nonexistent")


# ── MislabelDetector ──────────────────────────────────────────────────────────

class TestMislabelDetector:
    def test_knn_score_shape(self, embeddings_and_labels):
        emb, labels = embeddings_and_labels
        md = MislabelDetector(emb, labels, n_neighbors=5)
        scores = md.knn_label_quality()
        assert scores.shape == (len(labels),)

    def test_knn_score_range(self, embeddings_and_labels):
        emb, labels = embeddings_and_labels
        md = MislabelDetector(emb, labels, n_neighbors=5)
        scores = md.knn_label_quality()
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_combined_score_shape(self, embeddings_and_labels):
        emb, labels = embeddings_and_labels
        md = MislabelDetector(emb, labels, n_neighbors=5)
        scores = md.combined_score()
        assert scores.shape == (len(labels),)

    def test_rank_candidates_sorted(self, embeddings_and_labels):
        emb, labels = embeddings_and_labels
        md = MislabelDetector(emb, labels, n_neighbors=5)
        candidates = md.rank_candidates(top_k=20)
        mislabel_scores = [c["mislabel_score"] for c in candidates]
        assert mislabel_scores == sorted(mislabel_scores, reverse=True)

    def test_rank_candidates_has_required_keys(self, embeddings_and_labels):
        emb, labels = embeddings_and_labels
        md = MislabelDetector(emb, labels, n_neighbors=5)
        candidates = md.rank_candidates(top_k=5)
        for c in candidates:
            assert "index" in c
            assert "given_label" in c
            assert "suggested_label" in c
            assert "quality_score" in c

    def test_cluster_labels_get_high_quality(self):
        """Perfectly clustered data should score highly."""
        rng = np.random.default_rng(99)
        k, n_per_class, d = 4, 50, 32
        centers = np.eye(k, d)
        emb = np.vstack([centers[c] + 0.01 * rng.standard_normal((n_per_class, d)) for c in range(k)])
        labels = np.repeat(np.arange(k), n_per_class)
        md = MislabelDetector(emb, labels, n_neighbors=5)
        scores = md.knn_label_quality()
        assert scores.mean() > 0.9

    def test_from_cross_val_probs(self, pred_probs_and_labels):
        probs, labels = pred_probs_and_labels
        md = MislabelDetector.from_cross_val_probs(probs, labels)
        assert md.knn_label_quality().shape == (len(labels),)
