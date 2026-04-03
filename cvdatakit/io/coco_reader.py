"""COCO-format dataset reader with lazy image loading."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image


class COCODataset:
    """Thin wrapper around a COCO-format annotation file.

    Parameters
    ----------
    annotation_file:
        Path to the COCO JSON annotation file.
    image_dir:
        Root directory where images are stored.  When *None* image loading
        is disabled (statistics that need only annotations still work).
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: Optional[str | Path] = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir) if image_dir else None

        with open(self.annotation_file, "r") as fh:
            raw = json.load(fh)

        self._raw: Dict[str, Any] = raw

        # ── index structures ────────────────────────────────────────────────
        self.images: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in raw.get("images", [])
        }
        self.categories: Dict[int, Dict[str, Any]] = {
            cat["id"]: cat for cat in raw.get("categories", [])
        }
        self.annotations: List[Dict[str, Any]] = raw.get("annotations", [])

        # image_id → list[annotation]
        self._img2anns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in self.annotations:
            self._img2anns[ann["image_id"]].append(ann)

        # category_id → list[annotation]
        self._cat2anns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in self.annotations:
            self._cat2anns[ann["category_id"]].append(ann)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def num_annotations(self) -> int:
        return len(self.annotations)

    @property
    def category_names(self) -> List[str]:
        return [c["name"] for c in self.categories.values()]

    # ── accessors ───────────────────────────────────────────────────────────

    def get_annotations_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        return self._img2anns[image_id]

    def get_annotations_for_category(self, category_id: int) -> List[Dict[str, Any]]:
        return self._cat2anns[category_id]

    def get_image_ids_for_category(self, category_id: int) -> List[int]:
        return list({ann["image_id"] for ann in self._cat2anns[category_id]})

    def load_image(self, image_id: int) -> Image.Image:
        """Load a PIL image by its COCO *image_id*."""
        if self.image_dir is None:
            raise RuntimeError("image_dir was not set; cannot load images.")
        meta = self.images[image_id]
        path = self.image_dir / meta["file_name"]
        return Image.open(path).convert("RGB")

    def iter_images(
        self,
        *,
        with_annotations: bool = False,
    ) -> Iterator[Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]]:
        """Yield (image_meta, annotations_or_None) for every image."""
        for img_id, img_meta in self.images.items():
            anns = self._img2anns[img_id] if with_annotations else None
            yield img_meta, anns

    # ── bbox helpers ─────────────────────────────────────────────────────────

    def get_bboxes(self, image_id: int) -> np.ndarray:
        """Return (N, 4) array of [x, y, w, h] boxes for *image_id*."""
        anns = self._img2anns[image_id]
        if not anns:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([a["bbox"] for a in anns], dtype=np.float32)

    def get_labels(self, image_id: int) -> np.ndarray:
        """Return (N,) integer category-id array for *image_id*."""
        anns = self._img2anns[image_id]
        if not anns:
            return np.empty((0,), dtype=np.int64)
        return np.array([a["category_id"] for a in anns], dtype=np.int64)

    # ── convenience ──────────────────────────────────────────────────────────

    def class_counts(self) -> Dict[str, int]:
        """Return {category_name: annotation_count} mapping."""
        return {
            self.categories[cat_id]["name"]: len(anns)
            for cat_id, anns in self._cat2anns.items()
        }

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple["COCODataset", "COCODataset"]:
        """Random train/val split at the image level.

        Returns two in-memory *COCODataset* objects backed by temporary dicts.
        """
        rng = np.random.default_rng(seed)
        ids = list(self.images.keys())
        rng.shuffle(ids)
        cut = int(len(ids) * train_ratio)
        return (
            self._subset(ids[:cut]),
            self._subset(ids[cut:]),
        )

    def _subset(self, image_ids: List[int]) -> "COCODataset":
        id_set = set(image_ids)
        raw = {
            "images": [self.images[i] for i in image_ids],
            "categories": list(self.categories.values()),
            "annotations": [a for a in self.annotations if a["image_id"] in id_set],
        }
        obj = object.__new__(COCODataset)
        obj.annotation_file = self.annotation_file
        obj.image_dir = self.image_dir
        obj._raw = raw
        obj.images = {img["id"]: img for img in raw["images"]}
        obj.categories = self.categories
        obj.annotations = raw["annotations"]
        obj._img2anns = defaultdict(list)
        for ann in obj.annotations:
            obj._img2anns[ann["image_id"]].append(ann)
        obj._cat2anns = defaultdict(list)
        for ann in obj.annotations:
            obj._cat2anns[ann["category_id"]].append(ann)
        return obj

    def __repr__(self) -> str:
        return (
            f"COCODataset(images={self.num_images}, "
            f"categories={self.num_categories}, "
            f"annotations={self.num_annotations})"
        )
