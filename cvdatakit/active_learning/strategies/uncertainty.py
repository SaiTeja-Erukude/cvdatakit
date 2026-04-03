"""Uncertainty-based active-learning query strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy


class UncertaintyStrategy:
    """Select the most uncertain unlabelled samples for annotation.

    Supported scoring functions
    ---------------------------
    ``"entropy"``
        Shannon entropy of the predicted distribution.  Maximising entropy
        selects the samples where the model is most confused.
    ``"margin"``
        Difference between the two highest class probabilities.
        Low margin → high uncertainty.
    ``"least_confidence"``
        1 − max p(y|x).  Simplest uncertainty estimate.
    ``"bald"``
        Bayesian Active Learning by Disagreement (Houlsby et al., 2011).
        Requires *T* Monte-Carlo dropout forward passes supplied as a 3-D
        array (T, N, K).  Falls back to entropy when only point estimates
        are available.

    Parameters
    ----------
    scoring:
        One of the method names listed above.
    """

    VALID_METHODS = {"entropy", "margin", "least_confidence", "bald"}

    def __init__(self, scoring: str = "entropy") -> None:
        if scoring not in self.VALID_METHODS:
            raise ValueError(f"scoring must be one of {self.VALID_METHODS}")
        self.scoring = scoring

    # ── main API ─────────────────────────────────────────────────────────────

    def score(
        self,
        pred_probs: np.ndarray,
        *,
        mc_samples: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return (N,) uncertainty score for each unlabelled sample.

        Higher score → more uncertain → higher query priority.

        Parameters
        ----------
        pred_probs:
            (N, K) predicted class probabilities.
        mc_samples:
            (T, N, K) Monte-Carlo dropout samples, required for ``"bald"``.
        """
        pred_probs = np.asarray(pred_probs, dtype=np.float64)
        if pred_probs.ndim != 2:
            raise ValueError("pred_probs must be 2-D (N, K)")

        if self.scoring == "entropy":
            return self._entropy(pred_probs)
        elif self.scoring == "margin":
            return self._margin(pred_probs)
        elif self.scoring == "least_confidence":
            return self._least_confidence(pred_probs)
        elif self.scoring == "bald":
            if mc_samples is not None:
                return self._bald(pred_probs, np.asarray(mc_samples, dtype=np.float64))
            return self._entropy(pred_probs)
        raise ValueError(self.scoring)  # unreachable

    def query(
        self,
        pred_probs: np.ndarray,
        budget: int,
        *,
        mc_samples: Optional[np.ndarray] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return indices of the *budget* most uncertain samples.

        Parameters
        ----------
        pred_probs:
            (N, K) array covering the **unlabelled pool** only.
        budget:
            Number of samples to select.
        unlabeled_indices:
            Optional (N,) array mapping pool rows back to original dataset
            indices.  When provided the returned indices are in original space.
        """
        scores = self.score(pred_probs, mc_samples=mc_samples)
        top = np.argsort(scores)[::-1][:budget]
        if unlabeled_indices is not None:
            return np.asarray(unlabeled_indices)[top]
        return top

    def ranked(
        self,
        pred_probs: np.ndarray,
        *,
        mc_samples: Optional[np.ndarray] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Return all samples ranked by descending uncertainty."""
        scores = self.score(pred_probs, mc_samples=mc_samples)
        order = np.argsort(scores)[::-1]
        ids = (
            np.asarray(unlabeled_indices)[order]
            if unlabeled_indices is not None
            else order
        )
        return [
            {
                "index": int(ids[i]),
                "uncertainty_score": float(scores[order[i]]),
                "predicted_label": int(np.argmax(pred_probs[order[i]])),
                "max_prob": float(pred_probs[order[i]].max()),
            }
            for i in range(len(order))
        ]

    # ── scoring methods ───────────────────────────────────────────────────────

    @staticmethod
    def _entropy(p: np.ndarray) -> np.ndarray:
        return scipy_entropy(p, axis=1)

    @staticmethod
    def _margin(p: np.ndarray) -> np.ndarray:
        sorted_p = np.sort(p, axis=1)
        return 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])

    @staticmethod
    def _least_confidence(p: np.ndarray) -> np.ndarray:
        return 1.0 - p.max(axis=1)

    @staticmethod
    def _bald(mean_probs: np.ndarray, mc_samples: np.ndarray) -> np.ndarray:
        """BALD = H[y|x, D] − E_θ[H[y|x, θ]].

        Parameters
        ----------
        mean_probs: (N, K)
        mc_samples: (T, N, K)
        """
        # Re-compute mean probs from MC samples for consistency
        mc_mean = mc_samples.mean(axis=0)  # (N, K)
        mc_mean = np.clip(mc_mean, 1e-12, 1.0)
        mc_mean /= mc_mean.sum(axis=1, keepdims=True)

        h_mean = scipy_entropy(mc_mean, axis=1)  # H of expected: (N,)
        # Expected H of each sample: mean over T passes
        per_pass_h = scipy_entropy(
            np.clip(mc_samples, 1e-12, 1.0), axis=2
        )  # (T, N)
        mean_h = per_pass_h.mean(axis=0)  # (N,)
        # Clip to 0 to avoid tiny floating-point negatives
        return np.maximum(h_mean - mean_h, 0.0)
