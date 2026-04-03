"""Shared fixtures for the test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from cvdatakit.io.coco_reader import COCODataset


# ── COCO fixture factory ──────────────────────────────────────────────────────

def _make_coco_json(
    num_images: int = 20,
    num_categories: int = 5,
    anns_per_image: int = 3,
    image_size: Tuple[int, int] = (640, 480),
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    W, H = image_size
    categories = [
        {"id": i, "name": f"class_{i}", "supercategory": "object"}
        for i in range(num_categories)
    ]
    images = [
        {"id": i, "file_name": f"img_{i:04d}.jpg", "width": W, "height": H}
        for i in range(num_images)
    ]
    annotations = []
    ann_id = 0
    for img_id in range(num_images):
        for _ in range(anns_per_image):
            x = float(rng.integers(0, W - 50))
            y = float(rng.integers(0, H - 50))
            w = float(rng.integers(10, min(100, W - int(x))))
            h = float(rng.integers(10, min(100, H - int(y))))
            cat = int(rng.integers(0, num_categories))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {"images": images, "categories": categories, "annotations": annotations}


@pytest.fixture
def coco_file(tmp_path: Path) -> Path:
    """Write a synthetic COCO annotation file and return its path."""
    data = _make_coco_json()
    p = tmp_path / "annotations.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def coco_dataset(coco_file: Path) -> COCODataset:
    return COCODataset(coco_file)


@pytest.fixture
def pred_probs_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n, k = 100, 5
    raw = rng.dirichlet(np.ones(k), size=n)
    labels = rng.integers(0, k, size=n)
    return raw.astype(np.float32), labels.astype(np.int64)


@pytest.fixture
def embeddings_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n, d, k = 200, 64, 5
    # Generate cluster-like embeddings
    centers = rng.standard_normal((k, d))
    labels = rng.integers(0, k, size=n)
    emb = centers[labels] + 0.5 * rng.standard_normal((n, d))
    return emb.astype(np.float32), labels.astype(np.int64)
