from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MethodStats:
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    projection_density: float = 1.0
    extra_notes: str = ""


class ContinualMethod(Protocol):
    name: str
    stats: MethodStats

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        ...

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class NearestClassMean:
    name = "ncm"

    def __init__(self) -> None:
        self.prototypes: dict[int, np.ndarray] = {}
        self.stats = MethodStats(extra_notes="feature-space nearest class mean baseline")

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        start = time.perf_counter()
        x = l2_normalize(x.astype(np.float32))
        for cls in np.unique(y):
            self.prototypes[int(cls)] = x[y == cls].mean(axis=0)
        self.stats.train_seconds += time.perf_counter() - start

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        x = l2_normalize(x.astype(np.float32))
        classes = np.asarray([c for c in seen_classes if int(c) in self.prototypes], dtype=np.int64)
        proto = np.stack([self.prototypes[int(c)] for c in classes], axis=0)
        proto = l2_normalize(proto.astype(np.float32))
        pred = classes[np.argmax(x @ proto.T, axis=1)]
        self.stats.predict_seconds += time.perf_counter() - start
        return pred


class DenseRanPAC:
    name = "ranpac_dense"

    def __init__(self, feature_dim: int, proj_dim: int = 2048, ridge: float = 1e-2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal((feature_dim, proj_dim)) / np.sqrt(feature_dim)).astype(np.float32)
        self.proj_dim = proj_dim
        self.ridge = ridge
        self.num_classes = 0
        self.g = np.zeros((proj_dim, proj_dim), dtype=np.float32)
        self.q = np.zeros((proj_dim, 0), dtype=np.float32)
        self.weights = np.zeros((0, proj_dim), dtype=np.float32)
        self.stats = MethodStats(projection_density=1.0, extra_notes="RanPAC-style dense random projection + ridge readout")

    def _transform(self, x: np.ndarray) -> np.ndarray:
        z = np.maximum(x.astype(np.float32) @ self.proj, 0.0)
        return z

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        start = time.perf_counter()
        self.num_classes = int(np.max(seen_classes)) + 1
        if self.q.shape[1] < self.num_classes:
            new_q = np.zeros((self.proj_dim, self.num_classes), dtype=np.float32)
            new_q[:, : self.q.shape[1]] = self.q
            self.q = new_q
        z = self._transform(x)
        y_onehot = np.zeros((len(y), self.num_classes), dtype=np.float32)
        y_onehot[np.arange(len(y)), y.astype(np.int64)] = 1.0
        self.g += z.T @ z
        self.q += z.T @ y_onehot
        reg = self.ridge * np.eye(self.proj_dim, dtype=np.float32)
        self.weights = np.linalg.solve(self.g + reg, self.q).T.astype(np.float32)
        self.stats.train_seconds += time.perf_counter() - start

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        z = self._transform(x)
        logits = z @ self.weights.T
        unseen = np.ones(logits.shape[1], dtype=bool)
        unseen[seen_classes.astype(np.int64)] = False
        logits[:, unseen] = -np.inf
        pred = np.argmax(logits, axis=1).astype(np.int64)
        self.stats.predict_seconds += time.perf_counter() - start
        return pred


class SparseFlyCL:
    name = "flycl_sparse"

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int = 4096,
        fan_in: int = 16,
        seed: int = 0,
        whiten: bool = True,
    ) -> None:
        rng = np.random.default_rng(seed)
        rows = np.repeat(np.arange(feature_dim), fan_in)
        cols = rng.integers(0, proj_dim, size=feature_dim * fan_in)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=feature_dim * fan_in)
        self.proj = sparse.csr_matrix((signs, (rows, cols)), shape=(feature_dim, proj_dim), dtype=np.float32)
        self.proj_dim = proj_dim
        self.whiten = whiten
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.prototypes: dict[int, np.ndarray] = {}
        density = float(self.proj.nnz / (feature_dim * proj_dim))
        self.stats = MethodStats(projection_density=density, extra_notes="Fly-CL-style sparse Kenyon-cell projection + decorrelated prototypes")

    def _transform(self, x: np.ndarray, update_stats: bool = False) -> np.ndarray:
        z = np.maximum(x.astype(np.float32) @ self.proj, 0.0).astype(np.float32)
        if self.whiten:
            if update_stats or self.mean is None or self.std is None:
                self.mean = z.mean(axis=0, keepdims=True)
                self.std = z.std(axis=0, keepdims=True) + 1e-5
            z = (z - self.mean) / self.std
        return l2_normalize(z)

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        start = time.perf_counter()
        z = self._transform(x, update_stats=self.mean is None)
        for cls in np.unique(y):
            self.prototypes[int(cls)] = z[y == cls].mean(axis=0)
        self.stats.train_seconds += time.perf_counter() - start

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        z = self._transform(x)
        classes = np.asarray([c for c in seen_classes if int(c) in self.prototypes], dtype=np.int64)
        proto = np.stack([self.prototypes[int(c)] for c in classes], axis=0)
        proto = l2_normalize(proto.astype(np.float32))
        pred = classes[np.argmax(z @ proto.T, axis=1)]
        self.stats.predict_seconds += time.perf_counter() - start
        return pred


class AdaptiveSparseFlyCL:
    name = "flycl_adaptive"

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int = 4096,
        fan_in: int = 6,
        seed: int = 0,
        whiten: bool = True,
        top_k: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        rows = np.repeat(np.arange(feature_dim), fan_in)
        cols = rng.integers(0, proj_dim, size=feature_dim * fan_in)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=feature_dim * fan_in)
        self.proj = sparse.csr_matrix((signs, (rows, cols)), shape=(feature_dim, proj_dim), dtype=np.float32)
        self.proj_dim = proj_dim
        self.whiten = whiten
        self.top_k = top_k
        self.count = 0
        self.sum = np.zeros((1, proj_dim), dtype=np.float64)
        self.sumsq = np.zeros((1, proj_dim), dtype=np.float64)
        self.mean = np.zeros((1, proj_dim), dtype=np.float32)
        self.std = np.ones((1, proj_dim), dtype=np.float32)
        self.class_sums: dict[int, np.ndarray] = {}
        self.class_counts: dict[int, int] = {}
        self.prototypes: dict[int, np.ndarray] = {}
        density = float(self.proj.nnz / (feature_dim * proj_dim))
        notes = "Adaptive Fly-CL: cumulative whitening statistics + rebuilt class prototypes"
        if top_k is not None:
            notes += f"; WTA top_k={top_k}"
        self.stats = MethodStats(projection_density=density, extra_notes=notes)

    def _project(self, x: np.ndarray) -> np.ndarray:
        z = np.maximum(x.astype(np.float32) @ self.proj, 0.0).astype(np.float32)
        if self.top_k is not None and 0 < self.top_k < z.shape[1]:
            keep = np.argpartition(z, -self.top_k, axis=1)[:, -self.top_k :]
            mask = np.zeros_like(z, dtype=bool)
            rows = np.arange(z.shape[0])[:, None]
            mask[rows, keep] = True
            z = np.where(mask, z, 0.0).astype(np.float32)
        return z

    def _update_stats(self, z: np.ndarray) -> None:
        self.count += len(z)
        self.sum += z.sum(axis=0, keepdims=True, dtype=np.float64)
        self.sumsq += (z.astype(np.float64) ** 2).sum(axis=0, keepdims=True)
        self.mean = (self.sum / max(self.count, 1)).astype(np.float32)
        var = self.sumsq / max(self.count, 1) - self.mean.astype(np.float64) ** 2
        self.std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32) + 1e-5

    def _normalize_projected(self, z: np.ndarray) -> np.ndarray:
        if self.whiten:
            z = (z - self.mean) / self.std
        return l2_normalize(z.astype(np.float32))

    def _rebuild_prototypes(self) -> None:
        self.prototypes = {
            cls: self._normalize_projected(raw_sum / max(self.class_counts[cls], 1)).reshape(-1)
            for cls, raw_sum in self.class_sums.items()
        }

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        start = time.perf_counter()
        z_raw = self._project(x)
        self._update_stats(z_raw)
        for cls in np.unique(y):
            cls_int = int(cls)
            raw_sum = z_raw[y == cls].sum(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
            self.class_sums[cls_int] = self.class_sums.get(cls_int, np.zeros_like(raw_sum)) + raw_sum
            self.class_counts[cls_int] = self.class_counts.get(cls_int, 0) + int((y == cls).sum())
        self._rebuild_prototypes()
        self.stats.train_seconds += time.perf_counter() - start

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        z = self._normalize_projected(self._project(x))
        classes = np.asarray([c for c in seen_classes if int(c) in self.prototypes], dtype=np.int64)
        proto = np.stack([self.prototypes[int(c)] for c in classes], axis=0)
        proto = l2_normalize(proto.astype(np.float32))
        pred = classes[np.argmax(z @ proto.T, axis=1)]
        self.stats.predict_seconds += time.perf_counter() - start
        return pred


class _PromptHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(feature_dim))
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.normalize(x + self.prompt, dim=1))


class SABERFeaturePrompt:
    """Feature-space migration of SABER into a LibContinual-like task lifecycle.

    The original SABER paper studies prompt-based continual learning with frozen
    pretrained language models. This compact image-classification version keeps
    the same algorithmic ingredients: task-specific prompts, task-correlation
    selection, protected gradient subspaces, and constrained backward updates.
    """

    name = "saber_feature_prompt"

    def __init__(
        self,
        feature_dim: int,
        lr: float = 0.03,
        epochs: int = 80,
        batch_size: int = 256,
        subspace_rank: int = 8,
        projection_threshold: float = 0.08,
        cosine_threshold: float = 0.0,
        backward_lr: float = 0.01,
        backward_steps: int = 15,
        seed: int = 0,
        device: str | None = None,
        enable_backward_refinement: bool = True,
    ) -> None:
        torch.manual_seed(seed)
        self.feature_dim = feature_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.subspace_rank = subspace_rank
        self.projection_threshold = projection_threshold
        self.cosine_threshold = cosine_threshold
        self.backward_lr = backward_lr
        self.backward_steps = backward_steps
        self.enable_backward_refinement = enable_backward_refinement
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.modules: dict[int, _PromptHead] = {}
        self.class_maps: dict[int, dict[int, int]] = {}
        self.inverse_maps: dict[int, np.ndarray] = {}
        self.protected_subspaces: dict[int, torch.Tensor] = {}
        self.mean_gradients: dict[int, torch.Tensor] = {}
        self.task_id = -1
        self.stats = MethodStats(extra_notes="SABER-style selective backward refinement over frozen image features")

    def _loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool = True):
        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        ds = torch.utils.data.TensorDataset(x_t, y_t)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_single_task(self, task_id: int, x: np.ndarray, y_global: np.ndarray) -> list[torch.Tensor]:
        model = self.modules[task_id]
        class_map = self.class_maps[task_id]
        y_local = np.asarray([class_map[int(v)] for v in y_global], dtype=np.int64)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        gradients: list[torch.Tensor] = []
        model.train()
        for _ in range(self.epochs):
            for xb, yb in self._loader(x, y_local):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                gradients.append(model.prompt.grad.detach().flatten().cpu().clone())
                opt.step()
        return gradients

    def _store_subspace(self, task_id: int, gradients: list[torch.Tensor]) -> None:
        if not gradients:
            return
        g = torch.stack(gradients, dim=0)
        self.mean_gradients[task_id] = g.mean(dim=0).to(self.device)
        centered = g - g.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered.to(self.device), full_matrices=False)
        rank = min(self.subspace_rank, vh.shape[0])
        self.protected_subspaces[task_id] = vh[:rank].T.contiguous()

    def _select_prior_tasks(self, x_current: np.ndarray, y_current_global: np.ndarray) -> list[int]:
        selected: list[int] = []
        x_t = torch.from_numpy(x_current.astype(np.float32)).to(self.device)
        for prior_task, model in self.modules.items():
            if prior_task == self.task_id:
                continue
            model.eval()
            labels = np.asarray(
                [self.class_maps[prior_task].get(int(v), -1) for v in y_current_global],
                dtype=np.int64,
            )
            keep = labels >= 0
            if keep.sum() == 0:
                # If labels do not overlap, use entropy minimization as a
                # task-agnostic current-task signal, matching SABER's replay-free spirit.
                xb = x_t[: min(len(x_t), self.batch_size)]
                logits = model(xb)
                probs = logits.softmax(dim=1)
                loss = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            else:
                xb = x_t[keep]
                yb = torch.from_numpy(labels[keep]).to(self.device)
                loss = F.cross_entropy(model(xb), yb)
            grad = torch.autograd.grad(loss, model.prompt, retain_graph=False)[0].flatten()
            basis = self.protected_subspaces.get(prior_task)
            old_grad = self.mean_gradients.get(prior_task)
            if basis is None or old_grad is None:
                continue
            projection = basis @ (basis.T @ grad)
            projection_score = torch.norm(projection) / (torch.norm(grad) + 1e-8)
            cosine = F.cosine_similarity(old_grad, grad, dim=0)
            if projection_score.item() >= self.projection_threshold and cosine.item() > self.cosine_threshold:
                selected.append(prior_task)
        return selected

    def _backward_refine(self, selected: list[int], x_current: np.ndarray) -> None:
        if not self.enable_backward_refinement:
            return
        x_t = torch.from_numpy(x_current.astype(np.float32)).to(self.device)
        for prior_task in selected:
            model = self.modules[prior_task]
            basis = self.protected_subspaces[prior_task]
            model.train()
            for _ in range(self.backward_steps):
                xb = x_t[torch.randperm(len(x_t), device=self.device)[: min(self.batch_size, len(x_t))]]
                logits = model(xb)
                probs = logits.softmax(dim=1)
                loss = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
                grad = torch.autograd.grad(loss, model.prompt, retain_graph=False)[0].flatten()
                protected = basis @ (basis.T @ grad)
                non_interfering = grad - protected
                with torch.no_grad():
                    model.prompt -= self.backward_lr * non_interfering.view_as(model.prompt)

    def fit_task(self, x: np.ndarray, y: np.ndarray, seen_classes: np.ndarray) -> None:
        start = time.perf_counter()
        self.task_id += 1
        classes = np.unique(y).astype(np.int64)
        self.class_maps[self.task_id] = {int(c): i for i, c in enumerate(classes)}
        self.inverse_maps[self.task_id] = classes
        self.modules[self.task_id] = _PromptHead(self.feature_dim, len(classes)).to(self.device)
        gradients = self._train_single_task(self.task_id, x, y)
        self._store_subspace(self.task_id, gradients)
        selected = self._select_prior_tasks(x, y)
        self._backward_refine(selected, x)
        self.stats.extra_notes = f"SABER-style SBR; last selected prior tasks={selected}"
        self.stats.train_seconds += time.perf_counter() - start

    def predict(self, x: np.ndarray, seen_classes: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        class_set = set(int(c) for c in seen_classes)
        candidate_tasks = [
            task_id for task_id, inv in self.inverse_maps.items()
            if any(int(c) in class_set for c in inv)
        ]
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        best_scores = []
        best_labels = []
        with torch.no_grad():
            for task_id in candidate_tasks:
                model = self.modules[task_id]
                logits = model(x_t)
                probs = logits.softmax(dim=1)
                conf, pred_local = probs.max(dim=1)
                pred_global = torch.from_numpy(self.inverse_maps[task_id]).to(self.device)[pred_local]
                mask = torch.tensor([int(v) in class_set for v in pred_global.detach().cpu().numpy()], device=self.device)
                conf = torch.where(mask, conf, torch.full_like(conf, -1.0))
                best_scores.append(conf)
                best_labels.append(pred_global)
            score_stack = torch.stack(best_scores, dim=1)
            label_stack = torch.stack(best_labels, dim=1)
            best_task = score_stack.argmax(dim=1)
            pred = label_stack[torch.arange(len(x_t), device=self.device), best_task].cpu().numpy().astype(np.int64)
        self.stats.predict_seconds += time.perf_counter() - start
        return pred
