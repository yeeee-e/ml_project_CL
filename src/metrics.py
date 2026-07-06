from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ContinualMetrics:
    final_accuracy: float
    average_incremental_accuracy: float
    average_forgetting: float
    backward_transfer: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred) * 100.0)


def summarize_accuracy_matrix(acc_matrix: np.ndarray) -> ContinualMetrics:
    valid_rows = []
    for t in range(acc_matrix.shape[0]):
        valid = acc_matrix[t, : t + 1]
        valid_rows.append(float(np.mean(valid)))

    final_accuracy = float(np.mean(acc_matrix[-1, :]))
    average_incremental_accuracy = float(np.mean(valid_rows))

    forgetting_values = []
    for task_id in range(acc_matrix.shape[1] - 1):
        best_before_final = np.max(acc_matrix[task_id:-1, task_id])
        forgetting_values.append(best_before_final - acc_matrix[-1, task_id])
    average_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0

    backward_transfer_values = []
    for task_id in range(acc_matrix.shape[1] - 1):
        backward_transfer_values.append(acc_matrix[-1, task_id] - acc_matrix[task_id, task_id])
    backward_transfer = float(np.mean(backward_transfer_values)) if backward_transfer_values else 0.0

    return ContinualMetrics(
        final_accuracy=final_accuracy,
        average_incremental_accuracy=average_incremental_accuracy,
        average_forgetting=average_forgetting,
        backward_transfer=backward_transfer,
    )
