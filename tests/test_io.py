"""Tests for cvdatakit.io."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvdatakit.io.coco_reader import COCODataset
from cvdatakit.io.report import ReportGenerator, _escape


class TestCOCODataset:
    def test_basic_counts(self, coco_dataset):
        assert coco_dataset.num_images == 20
        assert coco_dataset.num_categories == 5
        assert coco_dataset.num_annotations == 60

    def test_category_names(self, coco_dataset):
        names = coco_dataset.category_names
        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)

    def test_get_annotations_for_image(self, coco_dataset):
        for img_id in list(coco_dataset.images.keys())[:3]:
            anns = coco_dataset.get_annotations_for_image(img_id)
            assert isinstance(anns, list)

    def test_get_bboxes_shape(self, coco_dataset):
        img_id = list(coco_dataset.images.keys())[0]
        boxes = coco_dataset.get_bboxes(img_id)
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4

    def test_get_labels_shape(self, coco_dataset):
        img_id = list(coco_dataset.images.keys())[0]
        labels = coco_dataset.get_labels(img_id)
        assert labels.ndim == 1

    def test_class_counts_sums_to_total(self, coco_dataset):
        counts = coco_dataset.class_counts()
        assert sum(counts.values()) == coco_dataset.num_annotations

    def test_split_sizes(self, coco_dataset):
        train, val = coco_dataset.split(train_ratio=0.8)
        assert train.num_images + val.num_images == coco_dataset.num_images

    def test_split_no_overlap(self, coco_dataset):
        train, val = coco_dataset.split()
        train_ids = set(train.images.keys())
        val_ids = set(val.images.keys())
        assert train_ids.isdisjoint(val_ids)

    def test_repr(self, coco_dataset):
        r = repr(coco_dataset)
        assert "COCODataset" in r
        assert "images=" in r

    def test_missing_image_dir_raises_on_load(self, coco_dataset):
        img_id = list(coco_dataset.images.keys())[0]
        with pytest.raises(RuntimeError, match="image_dir"):
            coco_dataset.load_image(img_id)

    def test_iter_images(self, coco_dataset):
        items = list(coco_dataset.iter_images())
        assert len(items) == coco_dataset.num_images

    def test_iter_images_with_annotations(self, coco_dataset):
        for img_meta, anns in coco_dataset.iter_images(with_annotations=True):
            assert anns is not None

    def test_get_image_ids_for_category(self, coco_dataset):
        cat_id = list(coco_dataset.categories.keys())[0]
        ids = coco_dataset.get_image_ids_for_category(cat_id)
        assert isinstance(ids, list)

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(Exception):
            COCODataset(p)

    def test_empty_dataset(self, tmp_path):
        data = {"images": [], "categories": [], "annotations": []}
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(data))
        ds = COCODataset(p)
        assert ds.num_images == 0
        assert ds.num_categories == 0
        assert ds.num_annotations == 0


class TestReportGenerator:
    def test_add_section_and_to_dict(self):
        rg = ReportGenerator("test_ds")
        rg.add_section("foo", {"a": 1, "b": [1, 2, 3]})
        d = rg.to_dict()
        assert d["dataset"] == "test_ds"
        assert "foo" in d["sections"]

    def test_save_json(self, tmp_path):
        rg = ReportGenerator("test")
        rg.add_section("stats", {"n": 10})
        p = rg.save_json(tmp_path / "report.json")
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert loaded["sections"]["stats"]["n"] == 10

    def test_save_html(self, tmp_path):
        rg = ReportGenerator("test")
        rg.add_section("stats", {"n": 10, "classes": ["a", "b"]})
        p = rg.save_html(tmp_path / "report.html")
        assert p.exists()
        html = p.read_text()
        assert "<html" in html
        assert "test" in html

    def test_html_escapes_special_chars(self, tmp_path):
        rg = ReportGenerator("<script>alert(1)</script>")
        rg.add_section("x", {"key": "<value>"})
        p = rg.save_html(tmp_path / "r.html")
        html = p.read_text()
        assert "<script>" not in html

    def test_numpy_serialisation(self, tmp_path):
        rg = ReportGenerator("np_test")
        rg.add_section("arr", {"values": np.array([1, 2, 3]), "scalar": np.int32(42)})
        p = rg.save_json(tmp_path / "np.json")
        loaded = json.loads(p.read_text())
        assert loaded["sections"]["arr"]["scalar"] == 42


def test_escape():
    assert _escape("<foo & 'bar'>") == "&lt;foo &amp; 'bar'&gt;"
