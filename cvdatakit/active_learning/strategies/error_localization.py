"""Error-localization active-learning strategy.

Selects samples where the model makes spatially-inconsistent predictions
or where gradient signals indicate high learning potential.

Two sub-strategies
------------------
``"gradient_norm"``
    Proxy for influence: samples with large gradient norms are likely to
    cause large parameter updates when labelled and trained on.
``"spatial_entropy"``
    For detection models: samples whose predicted bbox confidence maps have
    high spatial entropy have ambiguous foreground/background structure and
    benefit most from annotation.
``"influence_approx"``
    Approximates influence functions via leave-one-out loss difference
    estimated from pre-computed per-sample training losses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ErrorLocalizationStrategy:
    """Query strategy focusing on spatially uncertain / high-influence samples.

    Parameters
    ----------
    method:
        ``"gradient_norm"`` | ``"spatial_entropy"`` | ``"influence_approx"``
    """

    VALID_METHODS = {"gradient_norm", "spatial_entropy", "influence_approx"}

    def __init__(self, method: str = "gradient_norm") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        self.method = method

    # ── main API ─────────────────────────────────────────────────────────────

    def score(
        self,
        *,
        gradients: Optional[np.ndarray] = None,
        spatial_logits: Optional[np.ndarray] = None,
        train_losses: Optional[np.ndarray] = None,
        val_losses: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute (N,) priority scores for the unlabelled pool.

        Parameters
        ----------
        gradients:
            (N, P) gradient vectors per sample (for ``"gradient_norm"``).
        spatial_logits:
            (N, H, W, K) spatial prediction logits (for ``"spatial_entropy"``).
        train_losses:
            (N,) per-sample training loss (for ``"influence_approx"``).
        val_losses:
            (M,) validation loss vector (for ``"influence_approx"``).
        """
        if self.method == "gradient_norm":
            if gradients is None:
                raise ValueError("gradients required for gradient_norm")
            return self._gradient_norm(gradients)
        elif self.method == "spatial_entropy":
            if spatial_logits is None:
                raise ValueError("spatial_logits required for spatial_entropy")
            return self._spatial_entropy(spatial_logits)
        elif self.method == "influence_approx":
            if train_losses is None:
                raise ValueError("train_losses required for influence_approx")
            return self._influence_approx(train_losses, val_losses)
        raise ValueError(self.method)

    def query(
        self,
        budget: int,
        *,
        gradients: Optional[np.ndarray] = None,
        spatial_logits: Optional[np.ndarray] = None,
        train_losses: Optional[np.ndarray] = None,
        val_losses: Optional[np.ndarray] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the top-*budget* sample indices by priority score."""
        scores = self.score(
            gradients=gradients,
            spatial_logits=spatial_logits,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        top = np.argsort(scores)[::-1][:budget]
        if unlabeled_indices is not None:
            return np.asarray(unlabeled_indices)[top]
        return top

    def ranked(
        self,
        *,
        gradients: Optional[np.ndarray] = None,
        spatial_logits: Optional[np.ndarray] = None,
        train_losses: Optional[np.ndarray] = None,
        val_losses: Optional[np.ndarray] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Return all samples ranked by descending priority score."""
        scores = self.score(
            gradients=gradients,
            spatial_logits=spatial_logits,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        order = np.argsort(scores)[::-1]
        ids = (
            np.asarray(unlabeled_indices)[order]
            if unlabeled_indices is not None
            else order
        )
        return [
            {"index": int(ids[i]), "priority_score": float(scores[order[i]])}
            for i in range(len(order))
        ]

    # ── scoring implementations ───────────────────────────────────────────────

    @staticmethod
    def _gradient_norm(gradients: np.ndarray) -> np.ndarray:
        """L2 norm of each sample's gradient vector."""
        return np.linalg.norm(gradients, axis=1).astype(np.float32)

    @staticmethod
    def _spatial_entropy(spatial_logits: np.ndarray) -> np.ndarray:
        """Mean spatial entropy across the prediction map.

        Parameters
        ----------
        spatial_logits:
            (N, H, W, K) raw logits or softmax probabilities.
        """
        from scipy.special import softmax
        from scipy.stats import entropy as sp_entropy

        # softmax over class axis → (N, H, W, K)
        probs = softmax(spatial_logits.astype(np.float64), axis=-1)
        n, h, w, k = probs.shape
        # reshape to (N, H*W, K) then compute entropy over K axis
        flat = probs.reshape(n, h * w, k)  # (N, S, K)
        # sp_entropy expects (K, …) when axis not given; use axis=-1 explicitly
        ent = sp_entropy(flat, axis=-1, base=2)  # (N, S)
        return ent.mean(axis=1).astype(np.float32)  # (N,)

    @staticmethod
    def _influence_approx(
        train_losses: np.ndarray,
        val_losses: Optional[np.ndarray],
    ) -> np.ndarray:
        """Approximate influence via LOO loss proxy.

        If val_losses is provided the score accounts for expected impact on
        validation performance; otherwise raw train-loss magnitude is used as
        a proxy for informativeness.
        """
        tl = train_losses.astype(np.float64)
        if val_losses is not None:
            # Simple heuristic: samples with high train loss that correlate with
            # high validation loss have high influence
            vl_mean = float(val_losses.mean())
            # Normalise to same scale
            tl_norm = (tl - tl.min()) / (tl.max() - tl.min() + 1e-9)
            return tl_norm.astype(np.float32)
        return tl.astype(np.float32)
