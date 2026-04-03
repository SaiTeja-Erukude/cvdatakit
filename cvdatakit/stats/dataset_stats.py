"""Dataset-level statistics and class-imbalance metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy

from cvdatakit.io.coco_reader import COCODataset


class DatasetStats:
    """Compute and expose statistics for a :class:`~cvdatakit.io.COCODataset`.

    Parameters
    ----------
    dataset:
        A loaded :class:`~cvdatakit.io.COCODataset`.

    Examples
    --------
    >>> from cvdatakit.io import COCODataset
    >>> from cvdatakit.stats import DatasetStats
    >>> ds = COCODataset("instances_train2017.json")
    >>> stats = DatasetStats(ds)
    >>> print(stats.summary())
    """

    def __init__(self, dataset: COCODataset) -> None:
        self.dataset = dataset
        self._cache: Dict[str, Any] = {}

    # ── top-level summary ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict suitable for logging / reporting."""
        imb = self.class_imbalance()
        bbox = self.bbox_statistics()
        return {
            "num_images": self.dataset.num_images,
            "num_categories": self.dataset.num_categories,
            "num_annotations": self.dataset.num_annotations,
            "annotations_per_image": {
                "mean": float(np.mean(self._anns_per_image())),
                "std": float(np.std(self._anns_per_image())),
                "max": int(np.max(self._anns_per_image())),
                "min": int(np.min(self._anns_per_image())),
            },
            "class_imbalance": imb,
            "bbox_statistics": bbox,
            "class_counts": self.dataset.class_counts(),
        }

    # ── per-class counts and ranks ───────────────────────────────────────────

    def class_distribution(self) -> List[Dict[str, Any]]:
        """Sorted list of {name, count, fraction} per category."""
        counts = self.dataset.class_counts()
        total = max(sum(counts.values()), 1)
        rows = [
            {"name": name, "count": cnt, "fraction": cnt / total}
            for name, cnt in counts.items()
        ]
        return sorted(rows, key=lambda r: r["count"], reverse=True)

    def tail_categories(self, percentile: float = 10.0) -> List[str]:
        """Categories whose annotation count falls below *percentile*-th percentile."""
        dist = self.class_distribution()
        counts = np.array([r["count"] for r in dist])
        threshold = float(np.percentile(counts, percentile))
        return [r["name"] for r in dist if r["count"] <= threshold]

    # ── imbalance metrics ─────────────────────────────────────────────────────

    def class_imbalance(self) -> Dict[str, float]:
        """Return several imbalance metrics for the label distribution.

        Metrics
        -------
        gini:
            Gini coefficient (0 = perfect balance, 1 = extreme imbalance).
        entropy_bits:
            Shannon entropy in bits (higher = more balanced).
        max_entropy_bits:
            Maximum possible entropy for *K* classes.
        imbalance_ratio:
            count(most_common) / count(least_common).
        effective_num_classes:
            exp(entropy), the "effective" number of balanced classes.
        """
        counts = np.array(list(self.dataset.class_counts().values()), dtype=np.float64)
        if counts.sum() == 0:
            return {}
        p = counts / counts.sum()
        gini = _gini(p)
        ent = float(scipy_entropy(p, base=2))
        k = len(counts)
        return {
            "gini": round(gini, 6),
            "entropy_bits": round(ent, 6),
            "max_entropy_bits": round(np.log2(k) if k > 1 else 0.0, 6),
            "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 4),
            "effective_num_classes": round(float(np.exp(scipy_entropy(p))), 4),
        }

    # ── bounding-box statistics ───────────────────────────────────────────────

    def bbox_statistics(self) -> Dict[str, Any]:
        """Compute size / aspect-ratio statistics across all bounding boxes."""
        all_bboxes: List[np.ndarray] = []
        for img_id in self.dataset.images:
            bboxes = self.dataset.get_bboxes(img_id)
            if bboxes.size:
                all_bboxes.append(bboxes)

        if not all_bboxes:
            return {}

        boxes = np.concatenate(all_bboxes, axis=0)  # (N, 4) [x,y,w,h]
        w = boxes[:, 2]
        h = boxes[:, 3]
        areas = w * h
        aspect = np.where(h > 0, w / h, 0.0)

        img_areas = np.array(
            [
                (meta.get("width") or 1) * (meta.get("height") or 1)
                for meta in self.dataset.images.values()
            ],
            dtype=np.float64,
        )
        avg_img_area = float(img_areas.mean()) if img_areas.size else 1.0

        def _pct(arr: np.ndarray) -> Dict[str, float]:
            return {
                "min": float(arr.min()),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(arr.max()),
            }

        return {
            "total_boxes": int(len(boxes)),
            "area": _pct(areas),
            "width": _pct(w),
            "height": _pct(h),
            "aspect_ratio": _pct(aspect),
            "relative_area": _pct(areas / avg_img_area),
        }

    # ── image dimension statistics ────────────────────────────────────────────

    def image_size_statistics(self) -> Dict[str, Any]:
        """Width/height/aspect distribution of images in the annotation file."""
        widths, heights = [], []
        for meta in self.dataset.images.values():
            if "width" in meta:
                widths.append(meta["width"])
            if "height" in meta:
                heights.append(meta["height"])

        if not widths:
            return {"note": "no size metadata in annotation file"}

        widths_arr = np.array(widths, dtype=np.float64)
        heights_arr = np.array(heights, dtype=np.float64)
        aspects = widths_arr / np.maximum(heights_arr, 1)

        def _stat(arr: np.ndarray) -> Dict[str, float]:
            return {
                "min": float(arr.min()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "max": float(arr.max()),
            }

        return {
            "width": _stat(widths_arr),
            "height": _stat(heights_arr),
            "aspect_ratio": _stat(aspects),
            "num_unique_sizes": int(len({(w, h) for w, h in zip(widths, heights)})),
        }

    # ── annotation density ────────────────────────────────────────────────────

    def annotation_density(self) -> Dict[str, Any]:
        """Per-category: average annotations per image that contains it."""
        density = {}
        for cat_id, cat_meta in self.dataset.categories.items():
            image_ids = self.dataset.get_image_ids_for_category(cat_id)
            if not image_ids:
                density[cat_meta["name"]] = 0.0
                continue
            counts = [
                len(self.dataset.get_annotations_for_image(img_id))
                for img_id in image_ids
            ]
            density[cat_meta["name"]] = round(float(np.mean(counts)), 4)
        return density

    # ── co-occurrence ─────────────────────────────────────────────────────────

    def co_occurrence_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Symmetric (K × K) matrix: M[i,j] = number of images where both
        category i and category j appear together.
        """
        cat_ids = sorted(self.dataset.categories.keys())
        idx = {cid: i for i, cid in enumerate(cat_ids)}
        k = len(cat_ids)
        mat = np.zeros((k, k), dtype=np.int32)

        for img_id in self.dataset.images:
            anns = self.dataset.get_annotations_for_image(img_id)
            cats_in_img = list({a["category_id"] for a in anns if a["category_id"] in idx})
            for i, ci in enumerate(cats_in_img):
                for cj in cats_in_img[i:]:
                    mat[idx[ci], idx[cj]] += 1
                    if ci != cj:
                        mat[idx[cj], idx[ci]] += 1

        names = [self.dataset.categories[cid]["name"] for cid in cat_ids]
        return mat, names

    # ── private helpers ───────────────────────────────────────────────────────

    def _anns_per_image(self) -> np.ndarray:
        key = "_anns_per_image"
        if key not in self._cache:
            self._cache[key] = np.array(
                [len(self.dataset.get_annotations_for_image(i)) for i in self.dataset.images],
                dtype=np.int32,
            )
        return self._cache[key]


# ── helpers ───────────────────────────────────────────────────────────────────

def _gini(proportions: np.ndarray) -> float:
    """Gini coefficient from a 1-D array of non-negative values."""
    p = np.sort(proportions)
    n = len(p)
    if n == 0 or p.sum() == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2 * (indices * p).sum()) / (n * p.sum()) - (n + 1) / n)
