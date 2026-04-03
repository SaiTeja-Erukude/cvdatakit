"""Diversity-based active-learning query strategies.

Implemented strategies
----------------------
* **k-Center Greedy** (CoreSet): greedily select points that minimise the
  maximum distance from any unlabelled point to the nearest labelled point.
* **Cluster-Margin**: cluster the unlabelled pool and sample high-uncertainty
  representatives from each cluster.
* **MinMax**: maximise the minimum pairwise distance among selected samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class DiversityStrategy:
    """Select a diverse subset of the unlabelled pool for annotation.

    Parameters
    ----------
    method:
        ``"coreset"`` | ``"cluster_margin"`` | ``"minmax"``
    """

    VALID_METHODS = {"coreset", "cluster_margin", "minmax"}

    def __init__(self, method: str = "coreset") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        self.method = method

    # ── main API ─────────────────────────────────────────────────────────────

    def query(
        self,
        unlabeled_embeddings: np.ndarray,
        budget: int,
        *,
        labeled_embeddings: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        unlabeled_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return indices of *budget* diverse samples.

        Parameters
        ----------
        unlabeled_embeddings:
            (N_u, D) feature vectors for the unlabelled pool.
        budget:
            Number of samples to select.
        labeled_embeddings:
            (N_l, D) feature vectors for already-labelled samples.
            Used by ``"coreset"`` to initialise coverage.
        pred_probs:
            (N_u, K) predicted class probabilities.
            Required for ``"cluster_margin"``.
        unlabeled_indices:
            Optional (N_u,) array for index remapping to original dataset space.
        """
        emb = normalize(unlabeled_embeddings.astype(np.float32))
        budget = min(budget, len(emb))

        if self.method == "coreset":
            selected = self._coreset(emb, labeled_embeddings, budget)
        elif self.method == "cluster_margin":
            if pred_probs is None:
                raise ValueError("cluster_margin requires pred_probs")
            selected = self._cluster_margin(emb, pred_probs, budget)
        else:
            selected = self._minmax(emb, budget)

        if unlabeled_indices is not None:
            return np.asarray(unlabeled_indices)[selected]
        return selected

    # ── strategy implementations ──────────────────────────────────────────────

    def _coreset(
        self,
        emb: np.ndarray,
        labeled_emb: Optional[np.ndarray],
        budget: int,
    ) -> np.ndarray:
        """Greedy k-center (CoreSet) selection."""
        n = len(emb)
        if labeled_emb is not None and len(labeled_emb) > 0:
            labeled_emb = normalize(labeled_emb.astype(np.float32))
            # min distance from each unlabelled point to any labelled point
            # done in chunks to avoid O(N_u * N_l) memory explosion
            min_dist = _min_distances_chunked(emb, labeled_emb)
        else:
            # Start from a random seed
            min_dist = np.full(n, np.inf, dtype=np.float32)

        selected: List[int] = []
        for _ in range(budget):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            # Update distances using the newly selected point
            new_dists = _cosine_distances(emb, emb[idx : idx + 1]).ravel()
            min_dist = np.minimum(min_dist, new_dists)
        return np.array(selected, dtype=np.int64)

    def _cluster_margin(
        self,
        emb: np.ndarray,
        pred_probs: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Cluster pool, select the most uncertain sample per cluster."""
        n_clusters = min(budget, len(emb))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        cluster_ids = km.fit_predict(emb)

        # Uncertainty via margin
        sorted_p = np.sort(pred_probs, axis=1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        uncertainty = 1.0 - margin

        selected: List[int] = []
        for c in range(n_clusters):
            mask = cluster_ids == c
            if not mask.any():
                continue
            indices = np.where(mask)[0]
            best = indices[np.argmax(uncertainty[indices])]
            selected.append(int(best))
            if len(selected) >= budget:
                break
        return np.array(selected, dtype=np.int64)

    def _minmax(self, emb: np.ndarray, budget: int) -> np.ndarray:
        """Iteratively select the point farthest from the current selection."""
        n = len(emb)
        first = int(np.random.randint(n))
        selected = [first]
        min_dist = _cosine_distances(emb, emb[first : first + 1]).ravel()

        for _ in range(budget - 1):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            new_dists = _cosine_distances(emb, emb[idx : idx + 1]).ravel()
            min_dist = np.minimum(min_dist, new_dists)
        return np.array(selected, dtype=np.int64)


# ── vector distance helpers ───────────────────────────────────────────────────

def _cosine_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - similarity) between each row of A and each row of B.
    Returns (len(A), len(B)).
    """
    sim = A @ B.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _min_distances_chunked(
    query: np.ndarray,
    reference: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:
    """Min cosine distance from each query row to any reference row (chunked)."""
    n = len(query)
    min_dist = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, len(reference), chunk):
        block = reference[start : start + chunk]
        dists = _cosine_distances(query, block).min(axis=1)
        min_dist = np.minimum(min_dist, dists.astype(np.float32))
    return min_dist
