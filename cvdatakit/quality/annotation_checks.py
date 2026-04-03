"""Low-level annotation integrity checks (geometry, completeness, duplicates)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvdatakit.io.coco_reader import COCODataset


@dataclass
class AnnotationIssue:
    image_id: int
    annotation_id: Optional[int]
    issue_type: str
    severity: str  # "error" | "warning" | "info"
    details: str
    extra: Dict[str, Any] = field(default_factory=dict)


class AnnotationChecker:
    """Run a battery of geometric and structural checks on COCO annotations.

    Parameters
    ----------
    dataset:
        Loaded :class:`~cvdatakit.io.COCODataset`.
    min_bbox_area:
        Boxes whose pixel area is below this threshold trigger a warning.
    max_overlap_iou:
        Same-class boxes with IoU above this value are flagged as near-duplicates.
    """

    def __init__(
        self,
        dataset: COCODataset,
        min_bbox_area: float = 4.0,
        max_overlap_iou: float = 0.85,
    ) -> None:
        self.dataset = dataset
        self.min_bbox_area = min_bbox_area
        self.max_overlap_iou = max_overlap_iou

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self) -> List[AnnotationIssue]:
        """Run all checks and return a flat list of :class:`AnnotationIssue`."""
        issues: List[AnnotationIssue] = []
        for img_id, img_meta in self.dataset.images.items():
            anns = self.dataset.get_annotations_for_image(img_id)
            issues += self._check_image_bounds(img_meta, anns)
            issues += self._check_tiny_boxes(img_id, anns)
            issues += self._check_invalid_boxes(img_id, anns)
            issues += self._check_near_duplicates(img_id, anns)
            issues += self._check_unknown_categories(img_id, anns)
        return issues

    def summary(self) -> Dict[str, Any]:
        issues = self.run()
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for iss in issues:
            by_type[iss.issue_type] = by_type.get(iss.issue_type, 0) + 1
            by_severity[iss.severity] = by_severity.get(iss.severity, 0) + 1
        return {
            "total_issues": len(issues),
            "by_type": by_type,
            "by_severity": by_severity,
            "issues": [
                {
                    "image_id": i.image_id,
                    "annotation_id": i.annotation_id,
                    "issue_type": i.issue_type,
                    "severity": i.severity,
                    "details": i.details,
                }
                for i in issues
            ],
        }

    # ── individual checks ─────────────────────────────────────────────────────

    def _check_image_bounds(
        self,
        img_meta: Dict[str, Any],
        anns: List[Dict[str, Any]],
    ) -> List[AnnotationIssue]:
        issues = []
        W = img_meta.get("width")
        H = img_meta.get("height")
        if W is None or H is None:
            return []
        for ann in anns:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if x < 0 or y < 0 or x + w > W + 1 or y + h > H + 1:
                issues.append(
                    AnnotationIssue(
                        image_id=ann["image_id"],
                        annotation_id=ann["id"],
                        issue_type="out_of_bounds",
                        severity="error",
                        details=(
                            f"BBox [{x:.1f},{y:.1f},{w:.1f},{h:.1f}] "
                            f"exceeds image size {W}×{H}"
                        ),
                    )
                )
        return issues

    def _check_tiny_boxes(
        self,
        img_id: int,
        anns: List[Dict[str, Any]],
    ) -> List[AnnotationIssue]:
        issues = []
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]
            if 0 < area < self.min_bbox_area:
                issues.append(
                    AnnotationIssue(
                        image_id=img_id,
                        annotation_id=ann["id"],
                        issue_type="tiny_box",
                        severity="warning",
                        details=f"BBox area={area:.2f} px² < threshold {self.min_bbox_area}",
                    )
                )
        return issues

    def _check_invalid_boxes(
        self,
        img_id: int,
        anns: List[Dict[str, Any]],
    ) -> List[AnnotationIssue]:
        issues = []
        for ann in anns:
            bbox = ann.get("bbox")
            if bbox is None:
                issues.append(
                    AnnotationIssue(
                        image_id=img_id,
                        annotation_id=ann.get("id"),
                        issue_type="missing_bbox",
                        severity="error",
                        details="Annotation has no 'bbox' field",
                    )
                )
                continue
            if len(bbox) != 4:
                issues.append(
                    AnnotationIssue(
                        image_id=img_id,
                        annotation_id=ann.get("id"),
                        issue_type="malformed_bbox",
                        severity="error",
                        details=f"Expected 4-element bbox, got {len(bbox)}",
                    )
                )
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                issues.append(
                    AnnotationIssue(
                        image_id=img_id,
                        annotation_id=ann.get("id"),
                        issue_type="degenerate_bbox",
                        severity="error",
                        details=f"Non-positive w={w} or h={h}",
                    )
                )
        return issues

    def _check_near_duplicates(
        self,
        img_id: int,
        anns: List[Dict[str, Any]],
    ) -> List[AnnotationIssue]:
        issues = []
        if len(anns) < 2:
            return issues
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                a, b = anns[i], anns[j]
                if a.get("category_id") != b.get("category_id"):
                    continue
                iou = _bbox_iou(a.get("bbox", []), b.get("bbox", []))
                if iou >= self.max_overlap_iou:
                    issues.append(
                        AnnotationIssue(
                            image_id=img_id,
                            annotation_id=a["id"],
                            issue_type="near_duplicate",
                            severity="warning",
                            details=(
                                f"Annotations {a['id']} and {b['id']} "
                                f"same class, IoU={iou:.3f}"
                            ),
                            extra={"partner_id": b["id"], "iou": iou},
                        )
                    )
        return issues

    def _check_unknown_categories(
        self,
        img_id: int,
        anns: List[Dict[str, Any]],
    ) -> List[AnnotationIssue]:
        issues = []
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in self.dataset.categories:
                issues.append(
                    AnnotationIssue(
                        image_id=img_id,
                        annotation_id=ann.get("id"),
                        issue_type="unknown_category",
                        severity="error",
                        details=f"category_id={cat_id} not in categories list",
                    )
                )
        return issues


# ── geometry helpers ──────────────────────────────────────────────────────────

def _bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """IoU between two [x, y, w, h] bounding boxes."""
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0
