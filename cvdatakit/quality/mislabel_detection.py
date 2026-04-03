"""Mislabel detection via embedding-space nearest-neighbour analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class MislabelDetector:
    """Detect likely mislabelled samples using embedding-based heuristics.

    Intuition: a correctly labelled sample should be close in embedding space
    to other samples of the same class.  If its *k* nearest neighbours mostly
    belong to a different class, it is suspicious.

    Parameters
    ----------
    embeddings:
        (N, D) float array of per-sample feature embeddings.
    labels:
        (N,) integer array of given labels.
    n_neighbors:
        Number of neighbours used for the kNN graph (excluding self).
    metric:
        Distance metric passed to :class:`~sklearn.neighbors.NearestNeighbors`.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_neighbors: int = 10,
        metric: str = "cosine",
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (N, D)")
        if len(labels) != len(embeddings):
            raise ValueError("labels length must match embeddings")

        self.embeddings = normalize(embeddings.astype(np.float32))
        self.labels = labels.astype(np.int64)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._n = len(embeddings)

        self._knn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 to exclude self
            metric=metric,
            algorithm="brute",
            n_jobs=-1,
        ).fit(self.embeddings)

    # ── scores ────────────────────────────────────────────────────────────────

    def knn_label_quality(self) -> np.ndarray:
        """Return (N,) score ∈ [0, 1]: fraction of kNN neighbours sharing the
        same label.  Low score → potential mislabel.
        """
        distances, indices = self._knn.kneighbors(self.embeddings)
        # Remove self (first column since distances[i,0] ≈ 0)
        indices = indices[:, 1:]
        scores = np.zeros(self._n, dtype=np.float32)
        for i in range(self._n):
            neighbor_labels = self.labels[indices[i]]
            scores[i] = (neighbor_labels == self.labels[i]).mean()
        return scores

    def class_prototype_distance(self) -> np.ndarray:
        """Return (N,) distance of each sample to its class centroid.

        Larger distance relative to the class mean distance can indicate
        an outlier / mislabel.
        """
        k = int(self.labels.max()) + 1
        centroids = np.zeros((k, self.embeddings.shape[1]), dtype=np.float32)
        for c in range(k):
            mask = self.labels == c
            if mask.any():
                centroids[c] = self.embeddings[mask].mean(axis=0)

        centroids = normalize(centroids)
        dists = np.array(
            [1.0 - float(self.embeddings[i] @ centroids[self.labels[i]])
             for i in range(self._n)],
            dtype=np.float32,
        )
        return dists

    def combined_score(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> np.ndarray:
        """Weighted combination of knn quality and (inverted) prototype distance.

        Parameters
        ----------
        alpha:
            Weight for kNN label-agreement score.
        beta:
            Weight for class-prototype proximity score.

        Returns
        -------
        (N,) score ∈ [0, 1] – lower means more likely mislabelled.
        """
        knn_score = self.knn_label_quality()
        proto_dist = self.class_prototype_distance()
        # Normalise prototype distance to [0, 1]
        max_d = proto_dist.max()
        proto_score = 1.0 - (proto_dist / max(max_d, 1e-9))
        return alpha * knn_score + beta * proto_score

    # ── ranked candidates ─────────────────────────────────────────────────────

    def rank_candidates(
        self,
        top_k: Optional[int] = None,
        score_fn: str = "combined",
    ) -> List[Dict[str, Any]]:
        """Return samples ranked by ascending mislabel score (worst first).

        Parameters
        ----------
        top_k:
            Limit output to *top_k* candidates.
        score_fn:
            ``"knn"`` | ``"prototype"`` | ``"combined"``
        """
        if score_fn == "knn":
            scores = self.knn_label_quality()
        elif score_fn == "prototype":
            proto_dist = self.class_prototype_distance()
            max_d = proto_dist.max()
            scores = 1.0 - proto_dist / max(max_d, 1e-9)
        else:
            scores = self.combined_score()

        order = np.argsort(scores)
        if top_k is not None:
            order = order[:top_k]

        distances, indices = self._knn.kneighbors(self.embeddings)
        indices = indices[:, 1:]

        results = []
        for idx in order:
            neighbor_labels = self.labels[indices[idx]]
            label_counts = np.bincount(neighbor_labels)
            suggested = int(np.argmax(label_counts))
            results.append(
                {
                    "index": int(idx),
                    "given_label": int(self.labels[idx]),
                    "suggested_label": suggested,
                    "mislabel_score": float(1.0 - scores[idx]),
                    "quality_score": float(scores[idx]),
                    "neighbor_label_dist": label_counts.tolist(),
                }
            )
        return results

    def summary(self) -> Dict[str, Any]:
        scores = self.combined_score()
        flagged = (scores < 0.5).sum()
        return {
            "num_samples": int(self._n),
            "n_neighbors": self.n_neighbors,
            "mean_quality_score": float(scores.mean()),
            "flagged_count": int(flagged),
            "flagged_fraction": round(float(flagged) / self._n, 6),
            "top_10_candidates": self.rank_candidates(top_k=10),
        }

    # ── cross-validation mislabel detection ──────────────────────────────────

    @staticmethod
    def from_cross_val_probs(
        pred_probs: np.ndarray,
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        n_neighbors: int = 10,
    ) -> "MislabelDetector":
        """Construct a detector whose embeddings are *pred_probs* themselves.

        Useful when you have out-of-fold predicted probabilities but no
        separate feature embeddings.
        """
        emb = pred_probs if embeddings is None else embeddings
        return MislabelDetector(emb, labels, n_neighbors=n_neighbors, metric="cosine")
