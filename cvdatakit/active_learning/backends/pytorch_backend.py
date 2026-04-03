"""PyTorch model backend for active learning.

Requires ``torch`` and ``torchvision`` (install with ``pip install cvdatakit[torch]``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import ModelBackend


class PyTorchBackend(ModelBackend):
    """Wrap any ``torch.nn.Module`` for use with active-learning strategies.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` with a final linear/classification head.
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    transform:
        Optional callable applied to each PIL image before feeding to the model.
        If *None* a sensible default (resize 224 + ImageNet normalise) is built.
    embedding_layer:
        Attribute path of the penultimate feature layer, e.g. ``"avgpool"`` for
        standard torchvision ResNets.  Used by :meth:`get_embeddings`.

    Example
    -------
    >>> import torchvision.models as M
    >>> model = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    >>> backend = PyTorchBackend(model, device="cuda")
    >>> probs = backend.predict_proba(pil_image_list)
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        transform: Optional[Any] = None,
        embedding_layer: str = "avgpool",
    ) -> None:
        super().__init__(model, device)
        self.embedding_layer = embedding_layer
        self._transform = transform
        self._hook_output: Optional[Any] = None

    # ── ModelBackend interface ────────────────────────────────────────────────

    def predict_proba(
        self,
        images: Any,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        model = self._to_device()
        model.eval()
        all_probs: List[np.ndarray] = []
        transform = self._get_transform()

        with torch.no_grad():
            for batch in self._iter_batches(list(images), batch_size):
                tensors = torch.stack([transform(img) for img in batch]).to(self.device)
                logits = model(tensors)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def get_embeddings(
        self,
        images: Any,
        *,
        layer: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch

        layer_name = layer or self.embedding_layer
        model = self._to_device()
        model.eval()
        all_emb: List[np.ndarray] = []
        transform = self._get_transform()

        target_layer = self._resolve_layer(model, layer_name)
        hook_handle = target_layer.register_forward_hook(self._save_hook())

        try:
            with torch.no_grad():
                for batch in self._iter_batches(list(images), batch_size):
                    tensors = torch.stack([transform(img) for img in batch]).to(
                        self.device
                    )
                    model(tensors)
                    emb = self._hook_output
                    if emb.ndim > 2:
                        emb = emb.reshape(emb.shape[0], -1)
                    all_emb.append(emb)
        finally:
            hook_handle.remove()

        return np.concatenate(all_emb, axis=0)

    def compute_gradients(
        self,
        images: Any,
        labels: np.ndarray,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        model = self._to_device()
        model.train()
        transform = self._get_transform()
        all_grads: List[np.ndarray] = []
        image_list = list(images)
        label_list = labels.tolist()

        for i, batch_imgs in enumerate(self._iter_batches(image_list, batch_size)):
            batch_labels = label_list[i * batch_size : i * batch_size + len(batch_imgs)]
            tensors = torch.stack([transform(img) for img in batch_imgs]).to(self.device)
            label_t = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

            model.zero_grad()
            logits = model(tensors)
            loss = F.cross_entropy(logits, label_t, reduction="none")
            for j in range(len(batch_imgs)):
                model.zero_grad()
                loss[j].backward(retain_graph=j < len(batch_imgs) - 1)
                grad = np.concatenate(
                    [
                        p.grad.detach().cpu().numpy().ravel()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                )
                all_grads.append(grad)

        return np.stack(all_grads, axis=0)

    def mc_dropout_predict(
        self,
        images: Any,
        n_passes: int = 10,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """(T, N, K) stochastic predictions with dropout active."""
        import torch
        import torch.nn.functional as F

        model = self._to_device()
        # Keep dropout active; disable BN updates
        model.train()
        for m in model.modules():
            if "BatchNorm" in type(m).__name__:
                m.eval()

        transform = self._get_transform()
        image_list = list(images)
        passes: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(n_passes):
                all_probs: List[np.ndarray] = []
                for batch in self._iter_batches(image_list, batch_size):
                    tensors = torch.stack([transform(img) for img in batch]).to(
                        self.device
                    )
                    probs = F.softmax(model(tensors), dim=-1).cpu().numpy()
                    all_probs.append(probs)
                passes.append(np.concatenate(all_probs, axis=0))

        return np.stack(passes, axis=0)

    def train_one_epoch(
        self,
        images: Any,
        labels: np.ndarray,
        *,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        import torch
        import torch.nn.functional as F

        model = self._to_device()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        transform = self._get_transform()
        image_list = list(images)
        label_list = labels.tolist()

        total_loss = 0.0
        n_batches = 0

        for i, batch_imgs in enumerate(self._iter_batches(image_list, batch_size)):
            batch_labels = label_list[i * batch_size : i * batch_size + len(batch_imgs)]
            tensors = torch.stack([transform(img) for img in batch_imgs]).to(self.device)
            label_t = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

            optimizer.zero_grad()
            logits = model(tensors)
            loss = F.cross_entropy(logits, label_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # ── private helpers ───────────────────────────────────────────────────────

    def _to_device(self) -> Any:
        import torch

        return self.model.to(torch.device(self.device))

    def _get_transform(self) -> Any:
        if self._transform is not None:
            return self._transform
        try:
            from torchvision import transforms

            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for the default transform. "
                "Install with: pip install cvdatakit[torch]"
            ) from exc

    def _resolve_layer(self, model: Any, name: str) -> Any:
        """Resolve a dot-separated layer name to the actual nn.Module."""
        parts = name.split(".")
        layer = model
        for part in parts:
            layer = getattr(layer, part)
        return layer

    def _save_hook(self):
        def hook(module, input, output):
            import numpy as np

            out = output
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            self._hook_output = out

        return hook
