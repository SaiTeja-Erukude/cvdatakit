"""TensorFlow / Keras model backend for active learning.

Requires TensorFlow ≥ 2.13 (install with ``pip install cvdatakit[tensorflow]``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import ModelBackend


class TensorFlowBackend(ModelBackend):
    """Wrap a Keras ``Model`` for use with active-learning strategies.

    Parameters
    ----------
    model:
        A compiled ``tf.keras.Model``.
    device:
        TF device string, e.g. ``"cpu:0"``, ``"gpu:0"``.
    preprocess_fn:
        Optional callable applied to each numpy image (H, W, 3) before
        batching.  If *None*, images are resized to 224×224 and normalised
        to [0, 1].
    embedding_layer:
        Name or index of the feature-extraction layer used by
        :meth:`get_embeddings`.

    Example
    -------
    >>> import tensorflow as tf
    >>> base = tf.keras.applications.MobileNetV2(include_top=True)
    >>> backend = TensorFlowBackend(base)
    >>> probs = backend.predict_proba(numpy_image_list)
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu:0",
        preprocess_fn: Optional[Any] = None,
        embedding_layer: str | int = -2,
    ) -> None:
        super().__init__(model, device)
        self._preprocess_fn = preprocess_fn
        self.embedding_layer = embedding_layer

    # ── ModelBackend interface ────────────────────────────────────────────────

    def predict_proba(
        self,
        images: Any,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        import tensorflow as tf

        image_list = list(images)
        all_probs: List[np.ndarray] = []

        with tf.device(self.device):
            for batch in self._iter_batches(image_list, batch_size):
                tensor = self._preprocess_batch(batch)
                logits = self.model(tensor, training=False)
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def get_embeddings(
        self,
        images: Any,
        *,
        layer: Optional[str | int] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        import tensorflow as tf

        layer_ref = layer if layer is not None else self.embedding_layer
        image_list = list(images)

        # Build intermediate model
        if isinstance(layer_ref, int):
            out_layer = self.model.layers[layer_ref].output
        else:
            out_layer = self.model.get_layer(layer_ref).output

        emb_model = tf.keras.Model(inputs=self.model.input, outputs=out_layer)
        all_emb: List[np.ndarray] = []

        with tf.device(self.device):
            for batch in self._iter_batches(image_list, batch_size):
                tensor = self._preprocess_batch(batch)
                emb = emb_model(tensor, training=False).numpy()
                if emb.ndim > 2:
                    emb = emb.reshape(emb.shape[0], -1)
                all_emb.append(emb)

        return np.concatenate(all_emb, axis=0)

    def train_one_epoch(
        self,
        images: Any,
        labels: np.ndarray,
        *,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        import tensorflow as tf

        image_list = list(images)
        label_list = labels.tolist()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        total_loss = 0.0
        n_batches = 0

        with tf.device(self.device):
            for i, batch_imgs in enumerate(self._iter_batches(image_list, batch_size)):
                batch_labels = label_list[
                    i * batch_size : i * batch_size + len(batch_imgs)
                ]
                tensor = self._preprocess_batch(batch_imgs)
                label_t = tf.constant(batch_labels, dtype=tf.int32)

                with tf.GradientTape() as tape:
                    logits = self.model(tensor, training=True)
                    loss = loss_fn(label_t, logits)

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )
                total_loss += float(loss.numpy())
                n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # ── private helpers ───────────────────────────────────────────────────────

    def _preprocess_batch(self, images: List[Any]) -> Any:
        import tensorflow as tf

        arrays = []
        for img in images:
            arr = self._to_numpy(img)
            if self._preprocess_fn is not None:
                arr = self._preprocess_fn(arr)
            else:
                arr = tf.image.resize(arr, (224, 224)).numpy() / 255.0
            arrays.append(arr)
        return tf.stack(arrays, axis=0)

    @staticmethod
    def _to_numpy(img: Any) -> Any:
        """Convert PIL Image or numpy array to (H, W, 3) float32 numpy."""
        import numpy as np

        if hasattr(img, "numpy"):  # tf.Tensor
            return img.numpy().astype(np.float32)
        if hasattr(img, "__array__"):
            return np.asarray(img, dtype=np.float32)
        # PIL Image
        return np.array(img, dtype=np.float32)
