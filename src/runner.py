from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import numpy as np
import pandas as pd
import psutil
import yaml

from .datasets import make_class_order, make_incremental_tasks, mask_by_classes, ensure_dir
from .feature_cache import build_or_load_features
from .methods import AdaptiveSparseFlyCL, DenseRanPAC, NearestClassMean, SABERFeaturePrompt, SparseFlyCL
from .metrics import accuracy, summarize_accuracy_matrix


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_method(name: str, feature_dim: int, cfg: dict[str, Any]):
    kwargs = cfg.get("method_kwargs", {})
    if name == "ncm":
        return NearestClassMean()
    if name == "ranpac_dense":
        return DenseRanPAC(feature_dim=feature_dim, **kwargs)
    if name == "flycl_sparse":
        return SparseFlyCL(feature_dim=feature_dim, **kwargs)
    if name == "flycl_adaptive":
        return AdaptiveSparseFlyCL(feature_dim=feature_dim, **kwargs)
    if name == "saber_feature_prompt":
        return SABERFeaturePrompt(feature_dim=feature_dim, **kwargs)
    raise ValueError(f"unknown method: {name}")


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config["output_dir"])
    dataset_name = config.get("dataset_name", "cifar100")
    cache_path = build_or_load_features(
        data_dir=config["data_dir"],
        cache_dir=config["cache_dir"],
        dataset_name=dataset_name,
        backbone=config.get("backbone", "resnet18"),
        batch_size=int(config.get("feature_batch_size", 256)),
        num_workers=int(config.get("num_workers", 4)),
        force=bool(config.get("force_feature_rebuild", False)),
    )
    arrays = np.load(cache_path)
    x_train = arrays["train_features"].astype(np.float32)
    y_train = arrays["train_labels"].astype(np.int64)
    x_test = arrays["test_features"].astype(np.float32)
    y_test = arrays["test_labels"].astype(np.int64)

    total_cls_num = int(config["total_cls_num"])
    class_order = make_class_order(total_cls_num, config["seed"])
    tasks = make_incremental_tasks(class_order, config["init_cls_num"], config["inc_cls_num"])
    method = build_method(config["method"], x_train.shape[1], config)

    task_count = len(tasks)
    setting = config.get("setting", "task-agnostic")
    acc_matrix = np.full((task_count, task_count), np.nan, dtype=np.float32)
    wall_start = time.perf_counter()
    peak_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    per_step_rows = []
    for task in tasks:
        train_mask = mask_by_classes(y_train, task.new_classes)
        method.fit_task(x_train[train_mask], y_train[train_mask], task.seen_classes)

        for eval_task in tasks[: task.task_id + 1]:
            test_mask = mask_by_classes(y_test, eval_task.new_classes)
            eval_classes = eval_task.new_classes if setting == "task-aware" else task.seen_classes
            pred = method.predict(x_test[test_mask], eval_classes)
            acc = accuracy(y_test[test_mask], pred)
            acc_matrix[task.task_id, eval_task.task_id] = acc
            per_step_rows.append(
                {
                    "train_task": task.task_id,
                    "eval_task": eval_task.task_id,
                    "accuracy": acc,
                }
            )
        peak_rss_mb = max(peak_rss_mb, psutil.Process().memory_info().rss / (1024 * 1024))

    metrics = summarize_accuracy_matrix(acc_matrix)
    elapsed = time.perf_counter() - wall_start
    result = {
        "config": config,
        "method": config["method"],
        "backbone": config.get("backbone", "resnet18"),
        "seed": config["seed"],
        "class_order": class_order.tolist(),
        "metrics": metrics.to_dict(),
        "timing": {
            "wall_seconds": elapsed,
            "method_train_seconds": method.stats.train_seconds,
            "method_predict_seconds": method.stats.predict_seconds,
        },
        "resource": {
            "peak_rss_mb": peak_rss_mb,
            "projection_density": method.stats.projection_density,
        },
        "notes": method.stats.extra_notes,
    }

    stem = f"{config['experiment_name']}_{config['method']}_{config.get('backbone', 'resnet18')}_seed{config['seed']}"
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(per_step_rows).to_csv(output_dir / f"{stem}_per_step.csv", index=False)
    pd.DataFrame(acc_matrix).to_csv(output_dir / f"{stem}_acc_matrix.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["timing"], indent=2))


if __name__ == "__main__":
    main()
