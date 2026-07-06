from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IncrementalTask:
    task_id: int
    new_classes: np.ndarray
    seen_classes: np.ndarray


def make_class_order(num_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(num_classes).astype(np.int64)


def make_incremental_tasks(
    class_order: Iterable[int],
    init_cls: int,
    inc_cls: int,
) -> list[IncrementalTask]:
    order = np.asarray(list(class_order), dtype=np.int64)
    tasks: list[IncrementalTask] = []
    start = 0
    width = init_cls
    task_id = 0
    while start < len(order):
        end = min(start + width, len(order))
        new_classes = order[start:end]
        seen_classes = order[:end]
        tasks.append(
            IncrementalTask(
                task_id=task_id,
                new_classes=new_classes.copy(),
                seen_classes=seen_classes.copy(),
            )
        )
        start = end
        width = inc_cls
        task_id += 1
    return tasks


def mask_by_classes(labels: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    classes = np.asarray(list(classes), dtype=labels.dtype)
    return np.isin(labels, classes)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
