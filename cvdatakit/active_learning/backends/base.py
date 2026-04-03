"""Abstract base class for pluggable model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


class ModelBackend(ABC):
    """Interface that active-learning strategies use to interact with models.

    Concrete subclasses wrap PyTorch, TensorFlow, or any other framework.
    Only the methods required by a given strategy need to be implemented;
    others can raise :exc:`NotImplementedError`.

    Parameters
    ----------
    model:
        The underlying framework-specific model object.
    device:
        Device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"gpu:0"`` …).
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    # ── required interface ────────────────────────────────────────────────────

    @abstractmethod
    def predict_proba(
        self,
        images: Any,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return (N, K) predicted class probabilities.

        Parameters
        ----------
        images:
            Batch of images in whatever format the backend expects
            (list of PIL images, numpy arrays, dataset objects …).
        batch_size:
            Number of images to process per forward pass.
        """

    @abstractmethod
    def get_embeddings(
        self,
        images: Any,
        *,
        layer: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return (N, D) feature embeddings for *images*.

        Parameters
        ----------
        layer:
            Name of the intermediate layer to extract features from.
            When *None* the backend should use the penultimate layer.
        """

    # ── optional interface ────────────────────────────────────────────────────

    def compute_gradients(
        self,
        images: Any,
        labels: np.ndarray,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return (N, P) per-sample gradient vectors (flattened).

        Default implementation raises :exc:`NotImplementedError`.
        Override in backends that support gradient computation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_gradients"
        )

    def mc_dropout_predict(
        self,
        images: Any,
        n_passes: int = 10,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return (T, N, K) stochastic forward-pass predictions.

        Default implementation raises :exc:`NotImplementedError`.
        Override in backends that support MC dropout (typically PyTorch).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement mc_dropout_predict"
        )

    def train_one_epoch(
        self,
        images: Any,
        labels: np.ndarray,
        *,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        """Train the model for one epoch and return metrics.

        Returns a dict with at minimum ``{"loss": float}``.
        Override in backends that support in-loop retraining.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement train_one_epoch"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_batches(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
