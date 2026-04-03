"""Ready-made analysis recipe for COCO-format datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from cvdatakit.io.coco_reader import COCODataset
from cvdatakit.io.report import ReportGenerator
from cvdatakit.quality.annotation_checks import AnnotationChecker
from cvdatakit.stats.dataset_stats import DatasetStats


class COCORecipe:
    """Run the full cvdatakit analysis pipeline on a COCO-format dataset.

    Steps performed
    ---------------
    1. Load the annotation file.
    2. Compute dataset statistics (class distribution, imbalance, bbox stats).
    3. Run annotation-integrity checks (out-of-bounds, duplicates, tiny boxes …).
    4. Identify tail categories and co-occurrence structure.
    5. Optionally run mislabel / label-quality analysis when predicted
       probabilities or embeddings are supplied.
    6. Generate a JSON + HTML report.

    Parameters
    ----------
    annotation_file:
        Path to the COCO JSON annotation file.
    image_dir:
        Root directory of images.  Optional for statistics-only runs.
    report_dir:
        Directory where the generated reports will be saved.
        Defaults to the same directory as *annotation_file*.
    dataset_name:
        Human-readable dataset name used in the report title.

    Example
    -------
    >>> from cvdatakit.recipes import COCORecipe
    >>> recipe = COCORecipe("annotations/instances_train2017.json")
    >>> report = recipe.run()
    >>> print(report["stats"]["class_imbalance"])
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: Optional[str | Path] = None,
        report_dir: Optional[str | Path] = None,
        dataset_name: str = "COCO",
        min_bbox_area: float = 4.0,
        max_overlap_iou: float = 0.85,
        tail_percentile: float = 10.0,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir) if image_dir else None
        self.report_dir = (
            Path(report_dir) if report_dir else self.annotation_file.parent
        )
        self.dataset_name = dataset_name
        self.min_bbox_area = min_bbox_area
        self.max_overlap_iou = max_overlap_iou
        self.tail_percentile = tail_percentile

    def run(
        self,
        *,
        pred_probs: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        labels: Optional[Any] = None,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """Execute the analysis pipeline.

        Parameters
        ----------
        pred_probs:
            (N, K) predicted class probabilities for label-quality scoring.
        embeddings:
            (N, D) feature embeddings for mislabel detection.
        labels:
            (N,) integer ground-truth labels for quality / mislabel modules.
        save_report:
            Whether to write JSON and HTML report files to *report_dir*.

        Returns
        -------
        dict with keys ``"stats"``, ``"annotation_checks"``, and optionally
        ``"label_quality"`` / ``"mislabel_detection"``.
        """
        import numpy as np

        dataset = COCODataset(self.annotation_file, image_dir=self.image_dir)
        rg = ReportGenerator(dataset_name=self.dataset_name)

        # ── 1. statistics ─────────────────────────────────────────────────────
        stats_obj = DatasetStats(dataset)
        stats_dict = stats_obj.summary()
        stats_dict["tail_categories"] = stats_obj.tail_categories(self.tail_percentile)
        stats_dict["annotation_density"] = stats_obj.annotation_density()
        stats_dict["image_sizes"] = stats_obj.image_size_statistics()
        rg.add_section("stats", stats_dict)

        # ── 2. annotation checks ──────────────────────────────────────────────
        checker = AnnotationChecker(
            dataset,
            min_bbox_area=self.min_bbox_area,
            max_overlap_iou=self.max_overlap_iou,
        )
        check_dict = checker.summary()
        rg.add_section("annotation_checks", check_dict)

        result: Dict[str, Any] = {
            "stats": stats_dict,
            "annotation_checks": check_dict,
        }

        # ── 3. optional label quality ─────────────────────────────────────────
        if pred_probs is not None and labels is not None:
            from cvdatakit.quality.label_quality import LabelQualityScorer

            probs_arr = np.asarray(pred_probs)
            labels_arr = np.asarray(labels)
            lq = LabelQualityScorer(probs_arr, labels_arr)
            lq_dict = lq.summary()
            lq_dict["top_issues"] = lq.ranked_issues(top_k=50)
            rg.add_section("label_quality", lq_dict)
            result["label_quality"] = lq_dict

        # ── 4. optional mislabel detection ────────────────────────────────────
        if embeddings is not None and labels is not None:
            from cvdatakit.quality.mislabel_detection import MislabelDetector

            emb_arr = np.asarray(embeddings)
            labels_arr = np.asarray(labels)
            md = MislabelDetector(emb_arr, labels_arr)
            md_dict = md.summary()
            rg.add_section("mislabel_detection", md_dict)
            result["mislabel_detection"] = md_dict

        # ── 5. save report ────────────────────────────────────────────────────
        if save_report:
            stem = self.annotation_file.stem
            json_path = rg.save_json(self.report_dir / f"{stem}_report.json")
            html_path = rg.save_html(self.report_dir / f"{stem}_report.html")
            result["report_json"] = str(json_path)
            result["report_html"] = str(html_path)

        return result

    def recommend_resampling_weights(self) -> Dict[str, float]:
        """Compute per-class inverse-frequency weights for balanced sampling.

        Returns
        -------
        dict mapping category name → sampling weight.
        """
        dataset = COCODataset(self.annotation_file)
        counts = dataset.class_counts()
        total = sum(counts.values())
        n_classes = len(counts)
        weights = {}
        for name, cnt in counts.items():
            freq = cnt / max(total, 1)
            weights[name] = round(1.0 / (n_classes * max(freq, 1e-9)), 6)
        return weights

    def long_tail_report(self) -> Dict[str, Any]:
        """Focused report on long-tail class statistics."""
        dataset = COCODataset(self.annotation_file)
        stats_obj = DatasetStats(dataset)
        dist = stats_obj.class_distribution()
        imb = stats_obj.class_imbalance()
        tail = stats_obj.tail_categories(self.tail_percentile)
        return {
            "class_distribution": dist,
            "imbalance_metrics": imb,
            "tail_categories": tail,
            "tail_count": len(tail),
            "head_count": len(dist) - len(tail),
        }
