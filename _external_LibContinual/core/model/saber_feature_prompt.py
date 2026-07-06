"""
SABERFeaturePrompt

LibContinual migration of:
SABER: Turning Back Without Forgetting: Selective Backward Refinement for
Parameter-Efficient Continual Learning (ICML 2026 code release).

This implementation keeps the framework lifecycle used by LibContinual while
adapting SABER's selective backward refinement idea to image classification:
frozen backbone features, task-specific prompt vectors, task-local heads,
protected gradient subspaces, and constrained backward prompt refinement.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _PromptHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(feat_dim))
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.head(F.normalize(features + self.prompt, dim=1))


class SABERFeaturePrompt(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.subspace_rank = kwargs.get("subspace_rank", 8)
        self.projection_threshold = kwargs.get("projection_threshold", 0.08)
        self.cosine_threshold = kwargs.get("cosine_threshold", -0.05)
        self.backward_lr = kwargs.get("backward_lr", 0.01)
        self.backward_steps = kwargs.get("backward_steps", 15)
        self.enable_backward_refinement = kwargs.get("enable_backward_refinement", True)
        self.feat_dim = getattr(backbone, "feat_dim", kwargs.get("feat_dim", None))
        if self.feat_dim is None:
            self.feat_dim = kwargs.get("embd_dim", 512)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.task_modules = nn.ModuleList()
        self.task_classes = []
        self.mean_gradients = {}
        self.protected_subspaces = {}
        self.cur_task_id = -1
        self.current_gradients = []

    def _extract_features(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            out = out["features"]
        return out.detach()

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.cur_task_id = task_idx
        num_classes = self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self.task_modules.append(_PromptHead(self.feat_dim, num_classes).to(self.device))
        self.current_gradients = []
        try:
            class_names = train_loader.dataset.get_class_names()
            self.task_classes.append(list(range(task_idx * self.inc_cls_num, task_idx * self.inc_cls_num + len(class_names))))
        except Exception:
            start = 0 if task_idx == 0 else self.init_cls_num + (task_idx - 1) * self.inc_cls_num
            self.task_classes.append(list(range(start, start + num_classes)))

    def observe(self, data):
        x, y = data["image"].to(self.device), data["label"].to(self.device)
        start_cls = 0 if self.cur_task_id == 0 else self.init_cls_num + (self.cur_task_id - 1) * self.inc_cls_num
        y_local = y - start_cls
        features = self._extract_features(x)
        logits = self.task_modules[self.cur_task_id](features)
        loss = F.cross_entropy(logits, y_local)
        pred = logits.argmax(dim=1)
        acc = (pred == y_local).float().mean().item()
        loss.backward()
        grad = self.task_modules[self.cur_task_id].prompt.grad
        if grad is not None:
            self.current_gradients.append(grad.detach().flatten().cpu())
        return pred, acc, loss

    def _store_current_subspace(self):
        if len(self.current_gradients) == 0:
            return
        g = torch.stack(self.current_gradients, dim=0).to(self.device)
        self.mean_gradients[self.cur_task_id] = g.mean(dim=0)
        centered = g - g.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(self.subspace_rank, vh.shape[0])
        self.protected_subspaces[self.cur_task_id] = vh[:rank].T.contiguous()

    @torch.no_grad()
    def _select_prior_tasks(self, train_loader):
        if self.cur_task_id == 0:
            return []
        batch = next(iter(train_loader))
        x = batch["image"].to(self.device)
        features = self._extract_features(x)
        selected = []
        for task_id in range(self.cur_task_id):
            module = self.task_modules[task_id]
            with torch.enable_grad():
                logits = module(features)
                probs = logits.softmax(dim=1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
                grad = torch.autograd.grad(entropy, module.prompt, retain_graph=False)[0].flatten()
            basis = self.protected_subspaces.get(task_id)
            old_grad = self.mean_gradients.get(task_id)
            if basis is None or old_grad is None:
                continue
            projection = basis @ (basis.T @ grad)
            projection_score = torch.norm(projection) / (torch.norm(grad) + 1e-8)
            cosine = F.cosine_similarity(old_grad, grad, dim=0)
            if projection_score.item() >= self.projection_threshold and cosine.item() > self.cosine_threshold:
                selected.append(task_id)
        return selected

    def _backward_refine(self, selected, train_loader):
        if not self.enable_backward_refinement or len(selected) == 0:
            return
        batch = next(iter(train_loader))
        x = batch["image"].to(self.device)
        features = self._extract_features(x)
        for task_id in selected:
            module = self.task_modules[task_id]
            basis = self.protected_subspaces[task_id]
            for _ in range(self.backward_steps):
                logits = module(features)
                probs = logits.softmax(dim=1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
                grad = torch.autograd.grad(entropy, module.prompt, retain_graph=False)[0].flatten()
                protected = basis @ (basis.T @ grad)
                with torch.no_grad():
                    module.prompt -= self.backward_lr * (grad - protected).view_as(module.prompt)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self._store_current_subspace()
        selected = self._select_prior_tasks(train_loader)
        self._backward_refine(selected, train_loader)
        print(f"SABER selected prior tasks for backward refinement: {selected}")

    def inference(self, data, task_id=-1):
        x, y = data["image"].to(self.device), data["label"].to(self.device)
        features = self._extract_features(x)
        if task_id >= 0:
            logits = self.task_modules[task_id](features)
            local_pred = logits.argmax(dim=1)
            global_classes = torch.tensor(self.task_classes[task_id], device=self.device)
            pred = global_classes[local_pred]
        else:
            logits_all, labels_all = [], []
            for tid, module in enumerate(self.task_modules):
                logits = module(features)
                global_classes = torch.tensor(self.task_classes[tid], device=self.device)
                logits_all.append(logits)
                labels_all.append(global_classes)
            scores = torch.cat([v.softmax(dim=1) for v in logits_all], dim=1)
            labels = torch.cat(labels_all)
            pred = labels[scores.argmax(dim=1)]
        acc = (pred == y).float().mean().item()
        return pred, acc

    def get_parameters(self, config):
        return self.task_modules.parameters()
