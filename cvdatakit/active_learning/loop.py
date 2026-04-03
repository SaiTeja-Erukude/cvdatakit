"""Active-learning loop orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .backends.base import ModelBackend
from .strategies.diversity import DiversityStrategy
from .strategies.error_localization import ErrorLocalizationStrategy
from .strategies.uncertainty import UncertaintyStrategy

logger = logging.getLogger(__name__)

Strategy = Union[UncertaintyStrategy, DiversityStrategy, ErrorLocalizationStrategy]


@dataclass
class LoopConfig:
    """Configuration for :class:`ActiveLearningLoop`.

    Parameters
    ----------
    budget_per_round:
        Number of samples to query per active-learning round.
    max_rounds:
        Maximum number of rounds before the loop stops.
    min_quality_gain:
        If a quality metric callback is provided and improvement falls below
        this threshold for *patience* consecutive rounds, the loop stops early.
    patience:
        Consecutive rounds of insufficient improvement before early stopping.
    train_epochs_per_round:
        Epochs to train the model at the end of each round.
    seed:
        Random seed for reproducibility.
    """

    budget_per_round: int = 100
    max_rounds: int = 10
    min_quality_gain: float = 0.001
    patience: int = 3
    train_epochs_per_round: int = 1
    seed: int = 42


@dataclass
class RoundResult:
    round_idx: int
    queried_indices: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    labeled_pool_size: int = 0
    unlabeled_pool_size: int = 0


class ActiveLearningLoop:
    """Orchestrate a complete active-learning cycle.

    The loop manages two index pools (labeled / unlabeled), queries a
    strategy each round, optionally retrains the model, and records metrics.

    Parameters
    ----------
    backend:
        A :class:`~cvdatakit.active_learning.backends.ModelBackend` instance.
    strategy:
        A query strategy (uncertainty, diversity, or error-localization).
    images:
        Full image collection (list of PIL images, numpy arrays, or paths).
    labels:
        (N,) integer label array for *all* images (including unlabelled ones,
        which are masked during querying).
    config:
        Loop hyperparameters.
    initial_labeled_indices:
        Indices of samples in the initial labelled seed set.
        If *None* a random seed of ``budget_per_round`` samples is used.
    quality_metric_fn:
        Optional ``(backend, val_images, val_labels) → float`` callback that
        returns a scalar quality metric (e.g. accuracy) after each round.
    val_images / val_labels:
        Validation set passed to *quality_metric_fn*.

    Example
    -------
    >>> loop = ActiveLearningLoop(backend, strategy, images, labels)
    >>> history = loop.run()
    >>> print(history[-1].metrics)
    """

    def __init__(
        self,
        backend: ModelBackend,
        strategy: Strategy,
        images: Any,
        labels: np.ndarray,
        config: Optional[LoopConfig] = None,
        initial_labeled_indices: Optional[np.ndarray] = None,
        quality_metric_fn: Optional[Callable] = None,
        val_images: Optional[Any] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> None:
        self.backend = backend
        self.strategy = strategy
        self.images = list(images)
        self.labels = np.asarray(labels)
        self.config = config or LoopConfig()
        self.quality_metric_fn = quality_metric_fn
        self.val_images = val_images
        self.val_labels = val_labels

        rng = np.random.default_rng(self.config.seed)
        n = len(self.images)

        if initial_labeled_indices is not None:
            self._labeled = list(initial_labeled_indices)
        else:
            self._labeled = rng.choice(
                n, size=min(self.config.budget_per_round, n), replace=False
            ).tolist()

        self._unlabeled = [i for i in range(n) if i not in set(self._labeled)]
        self._history: List[RoundResult] = []

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self) -> List[RoundResult]:
        """Execute the full active-learning loop.

        Returns the round-by-round history as a list of :class:`RoundResult`.
        """
        no_gain_streak = 0
        prev_quality: Optional[float] = None

        for round_idx in range(self.config.max_rounds):
            if not self._unlabeled:
                logger.info("Unlabeled pool exhausted – stopping.")
                break

            logger.info(
                "Round %d/%d | labeled=%d | unlabeled=%d",
                round_idx + 1,
                self.config.max_rounds,
                len(self._labeled),
                len(self._unlabeled),
            )

            queried = self._query_round()
            self._move_to_labeled(queried)

            # Optionally retrain
            train_metrics = self._maybe_train()

            # Optionally evaluate
            quality_metrics = self._maybe_evaluate()
            all_metrics = {**train_metrics, **quality_metrics}

            result = RoundResult(
                round_idx=round_idx,
                queried_indices=queried,
                metrics=all_metrics,
                labeled_pool_size=len(self._labeled),
                unlabeled_pool_size=len(self._unlabeled),
            )
            self._history.append(result)

            # Early stopping
            if "quality" in quality_metrics and prev_quality is not None:
                gain = quality_metrics["quality"] - prev_quality
                if gain < self.config.min_quality_gain:
                    no_gain_streak += 1
                    logger.info("Low quality gain (%.5f), streak=%d", gain, no_gain_streak)
                else:
                    no_gain_streak = 0
                prev_quality = quality_metrics["quality"]
            elif "quality" in quality_metrics:
                prev_quality = quality_metrics["quality"]

            if no_gain_streak >= self.config.patience:
                logger.info("Early stopping triggered after round %d", round_idx + 1)
                break

        return self._history

    # ── helpers ───────────────────────────────────────────────────────────────

    def _query_round(self) -> np.ndarray:
        unlabeled_images = [self.images[i] for i in self._unlabeled]
        unlabeled_idx = np.array(self._unlabeled)

        budget = min(self.config.budget_per_round, len(unlabeled_idx))

        if isinstance(self.strategy, UncertaintyStrategy):
            probs = self.backend.predict_proba(unlabeled_images)
            return self.strategy.query(probs, budget, unlabeled_indices=unlabeled_idx)

        elif isinstance(self.strategy, DiversityStrategy):
            emb_unlabeled = self.backend.get_embeddings(unlabeled_images)
            labeled_images = [self.images[i] for i in self._labeled]
            emb_labeled = (
                self.backend.get_embeddings(labeled_images) if labeled_images else None
            )
            return self.strategy.query(
                emb_unlabeled,
                budget,
                labeled_embeddings=emb_labeled,
                unlabeled_indices=unlabeled_idx,
            )

        elif isinstance(self.strategy, ErrorLocalizationStrategy):
            if self.strategy.method == "gradient_norm":
                grads = self.backend.compute_gradients(
                    unlabeled_images,
                    self.labels[unlabeled_idx],
                )
                return self.strategy.query(budget, gradients=grads, unlabeled_indices=unlabeled_idx)
            else:
                # Fall back to loss-proxy using predicted probability of argmax
                probs = self.backend.predict_proba(unlabeled_images)
                # Use 1 - max_prob as proxy for train-loss
                proxy_loss = 1.0 - probs.max(axis=1)
                return self.strategy.query(
                    budget,
                    train_losses=proxy_loss,
                    unlabeled_indices=unlabeled_idx,
                )

        raise TypeError(f"Unsupported strategy type: {type(self.strategy)}")

    def _move_to_labeled(self, queried: np.ndarray) -> None:
        queried_set = set(queried.tolist())
        self._labeled.extend(queried_set)
        self._unlabeled = [i for i in self._unlabeled if i not in queried_set]

    def _maybe_train(self) -> Dict[str, float]:
        try:
            labeled_images = [self.images[i] for i in self._labeled]
            labeled_labels = self.labels[self._labeled]
            metrics = self.backend.train_one_epoch(
                labeled_images,
                labeled_labels,
            )
            return {f"train_{k}": v for k, v in metrics.items()}
        except NotImplementedError:
            return {}

    def _maybe_evaluate(self) -> Dict[str, float]:
        if self.quality_metric_fn is None or self.val_images is None:
            return {}
        try:
            score = self.quality_metric_fn(
                self.backend, self.val_images, self.val_labels
            )
            return {"quality": float(score)}
        except Exception as exc:
            logger.warning("quality_metric_fn raised: %s", exc)
            return {}

    # ── inspection ─────────────────────────────────────────────────────────────

    @property
    def labeled_indices(self) -> List[int]:
        return list(self._labeled)

    @property
    def unlabeled_indices(self) -> List[int]:
        return list(self._unlabeled)

    @property
    def history(self) -> List[RoundResult]:
        return list(self._history)

    def summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "round": r.round_idx + 1,
                "queried": len(r.queried_indices),
                "labeled_pool": r.labeled_pool_size,
                "unlabeled_pool": r.unlabeled_pool_size,
                **r.metrics,
            }
            for r in self._history
        ]
