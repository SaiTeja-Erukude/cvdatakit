"""Label-quality scoring using predicted class probabilities (Confident Learning)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy


class LabelQualityScorer:
    """Assign a quality score ∈ [0, 1] to each labeled example.

    The score reflects how consistent the given label is with the model's
    predicted class probabilities (higher = cleaner label).

    The implementation follows the *Confident Learning* paradigm
    (Northcutt et al., 2021) adapted for object-level classification.

    Parameters
    ----------
    pred_probs:
        (N, K) array of predicted class probabilities from a trained model.
    labels:
        (N,) integer array of given (possibly noisy) labels in [0, K).
    """

    def __init__(self, pred_probs: np.ndarray, labels: np.ndarray) -> None:
        if pred_probs.ndim != 2:
            raise ValueError("pred_probs must be 2-D (N, K)")
        if labels.ndim != 1 or len(labels) != len(pred_probs):
            raise ValueError("labels must be 1-D with length N")
        self.pred_probs = pred_probs.astype(np.float64)
        self.labels = labels.astype(np.int64)
        self._n, self._k = pred_probs.shape

    # ── main API ─────────────────────────────────────────────────────────────

    def quality_scores(self, method: str = "normalized_margin") -> np.ndarray:
        """Return (N,) quality score array.

        Parameters
        ----------
        method:
            ``"normalized_margin"`` – margin between given-class prob and
            argmax prob, normalised to [0, 1].
            ``"self_confidence"`` – raw predicted probability for the given label.
            ``"entropy_weighted"`` – self-confidence weighted by inverse entropy.
        """
        if method == "normalized_margin":
            return self._normalized_margin()
        elif method == "self_confidence":
            return self._self_confidence()
        elif method == "entropy_weighted":
            return self._entropy_weighted()
        else:
            raise ValueError(f"Unknown method: {method!r}")

    def find_label_issues(
        self,
        threshold: float = 0.5,
        method: str = "normalized_margin",
    ) -> np.ndarray:
        """Return boolean mask of samples with quality score < *threshold*."""
        scores = self.quality_scores(method)
        return scores < threshold

    def ranked_issues(
        self,
        method: str = "normalized_margin",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return samples ranked by ascending quality score (worst first).

        Returns a list of dicts with keys: ``index``, ``given_label``,
        ``predicted_label``, ``quality_score``, ``confidence_in_given``.
        """
        scores = self.quality_scores(method)
        order = np.argsort(scores)
        if top_k is not None:
            order = order[:top_k]
        return [
            {
                "index": int(i),
                "given_label": int(self.labels[i]),
                "predicted_label": int(np.argmax(self.pred_probs[i])),
                "quality_score": float(scores[i]),
                "confidence_in_given": float(self.pred_probs[i, self.labels[i]]),
            }
            for i in order
        ]

    def confusion_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate K×K joint label-noise matrix.

        Returns
        -------
        joint:
            (K, K) matrix where joint[s, t] ≈ P(given=s, true=t).
        marginal_true:
            (K,) estimated marginal distribution of true labels.
        """
        thresholds = self._per_class_thresholds()
        # Build confident-joint counts
        cj = np.zeros((self._k, self._k), dtype=np.int32)
        for i in range(self._n):
            s = int(self.labels[i])
            for t in range(self._k):
                if self.pred_probs[i, t] >= thresholds[t]:
                    cj[s, t] += 1

        total = cj.sum()
        joint = cj / max(total, 1)
        marginal_true = joint.sum(axis=0)
        return joint, marginal_true

    def summary(self, method: str = "normalized_margin") -> Dict[str, Any]:
        scores = self.quality_scores(method)
        mask = scores < 0.5
        joint, _ = self.confusion_matrix()
        off_diag_mass = float(1 - np.diag(joint).sum())
        return {
            "method": method,
            "num_samples": int(self._n),
            "num_classes": int(self._k),
            "mean_quality_score": float(scores.mean()),
            "estimated_error_rate": round(off_diag_mass, 6),
            "flagged_count": int(mask.sum()),
            "flagged_fraction": round(float(mask.sum()) / self._n, 6),
        }

    # ── private scorers ───────────────────────────────────────────────────────

    def _self_confidence(self) -> np.ndarray:
        return self.pred_probs[np.arange(self._n), self.labels]

    def _normalized_margin(self) -> np.ndarray:
        self_conf = self._self_confidence()
        top_conf = self.pred_probs.max(axis=1)
        # margin in [-1, 1], shift to [0, 1]
        margin = self_conf - top_conf
        return (margin + 1.0) / 2.0

    def _entropy_weighted(self) -> np.ndarray:
        sc = self._self_confidence()
        ent = scipy_entropy(self.pred_probs, axis=1) / np.log(self._k + 1e-9)
        # low entropy → model is confident → weight matters more
        return sc * (1.0 - ent)

    def _per_class_thresholds(self) -> np.ndarray:
        """Average predicted probability for each class k among samples labeled k."""
        thresholds = np.zeros(self._k, dtype=np.float64)
        for k in range(self._k):
            mask = self.labels == k
            if mask.any():
                thresholds[k] = self.pred_probs[mask, k].mean()
            else:
                thresholds[k] = 0.5
        return thresholds
