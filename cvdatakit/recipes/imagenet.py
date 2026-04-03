"""Recipe helpers for ImageNet-style flat-folder datasets.

ImageNet does not use COCO format; images are organised as::

    root/
        synset_id_1/
            img1.JPEG
            img2.JPEG
        synset_id_2/
            ...

This module converts such a structure to an in-memory COCO-compatible
:class:`~cvdatakit.io.COCODataset` and exposes the same analysis pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvdatakit.io.coco_reader import COCODataset
from cvdatakit.io.report import ReportGenerator
from cvdatakit.stats.dataset_stats import DatasetStats


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ImageNetRecipe:
    """Analyse an ImageNet-style flat-folder dataset.

    Parameters
    ----------
    root:
        Dataset root directory (each subdirectory = one class).
    report_dir:
        Directory to write the report to.  Defaults to *root*.
    dataset_name:
        Name shown in the report.
    max_images_per_class:
        Cap on images loaded per class (useful for huge datasets).

    Example
    -------
    >>> from cvdatakit.recipes import ImageNetRecipe
    >>> recipe = ImageNetRecipe("/data/imagenet/val")
    >>> report = recipe.run()
    """

    def __init__(
        self,
        root: str | Path,
        report_dir: Optional[str | Path] = None,
        dataset_name: str = "ImageNet",
        max_images_per_class: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.report_dir = Path(report_dir) if report_dir else self.root
        self.dataset_name = dataset_name
        self.max_images_per_class = max_images_per_class

    # ── public API ────────────────────────────────────────────────────────────

    def scan(self) -> Tuple[List[Path], List[int], List[str]]:
        """Scan the root directory and return (paths, labels, class_names).

        Returns
        -------
        paths:
            Absolute paths to image files.
        labels:
            Integer class index for each image.
        class_names:
            Sorted list of class names (synset folder names).
        """
        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        class_names = [d.name for d in class_dirs]
        paths: List[Path] = []
        labels: List[int] = []

        for cls_idx, cls_dir in enumerate(class_dirs):
            imgs = [
                f
                for f in cls_dir.iterdir()
                if f.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if self.max_images_per_class is not None:
                imgs = imgs[: self.max_images_per_class]
            paths.extend(imgs)
            labels.extend([cls_idx] * len(imgs))

        return paths, labels, class_names

    def to_coco_dataset(self) -> COCODataset:
        """Convert the folder structure to an in-memory :class:`COCODataset`."""
        paths, labels, class_names = self.scan()

        categories = [
            {"id": i, "name": name, "supercategory": ""}
            for i, name in enumerate(class_names)
        ]
        images = [
            {
                "id": i,
                "file_name": str(p.relative_to(self.root)),
                "width": None,
                "height": None,
            }
            for i, p in enumerate(paths)
        ]
        # For classification datasets we create a dummy "full-image" bbox
        annotations = [
            {
                "id": i,
                "image_id": i,
                "category_id": labels[i],
                "bbox": [0, 0, 1, 1],  # placeholder
                "area": 1,
                "iscrowd": 0,
            }
            for i in range(len(paths))
        ]

        raw = {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

        ds = object.__new__(COCODataset)
        from collections import defaultdict

        ds.annotation_file = self.root / "__imagenet_virtual__.json"
        ds.image_dir = self.root
        ds._raw = raw
        ds.images = {img["id"]: img for img in raw["images"]}
        ds.categories = {cat["id"]: cat for cat in raw["categories"]}
        ds.annotations = raw["annotations"]
        ds._img2anns = defaultdict(list)
        for ann in ds.annotations:
            ds._img2anns[ann["image_id"]].append(ann)
        ds._cat2anns = defaultdict(list)
        for ann in ds.annotations:
            ds._cat2anns[ann["category_id"]].append(ann)
        return ds

    def run(self, *, save_report: bool = True) -> Dict[str, Any]:
        """Run statistics and imbalance analysis on the folder dataset."""
        dataset = self.to_coco_dataset()
        rg = ReportGenerator(dataset_name=self.dataset_name)

        stats_obj = DatasetStats(dataset)
        stats_dict = stats_obj.summary()
        stats_dict["tail_categories"] = stats_obj.tail_categories()
        rg.add_section("stats", stats_dict)

        result: Dict[str, Any] = {"stats": stats_dict}

        if save_report:
            json_path = rg.save_json(self.report_dir / "imagenet_report.json")
            html_path = rg.save_html(self.report_dir / "imagenet_report.html")
            result["report_json"] = str(json_path)
            result["report_html"] = str(html_path)

        return result

    def class_imbalance_summary(self) -> Dict[str, Any]:
        """Quick imbalance summary without a full run."""
        dataset = self.to_coco_dataset()
        stats_obj = DatasetStats(dataset)
        return {
            "class_imbalance": stats_obj.class_imbalance(),
            "class_distribution": stats_obj.class_distribution(),
            "tail_categories": stats_obj.tail_categories(),
        }

    def recommend_oversampling(self, target_count: Optional[int] = None) -> Dict[str, int]:
        """For each class, recommend how many extra samples to generate/duplicate.

        Parameters
        ----------
        target_count:
            Desired annotation count per class.  Defaults to the count of the
            most common class (i.e. upsample minority classes to match the head).

        Returns
        -------
        dict mapping class_name → extra samples needed.
        """
        dataset = self.to_coco_dataset()
        counts = dataset.class_counts()
        if not counts:
            return {}
        max_count = target_count or max(counts.values())
        return {
            name: max(0, max_count - cnt)
            for name, cnt in counts.items()
        }
