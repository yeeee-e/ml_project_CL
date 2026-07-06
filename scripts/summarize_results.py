from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="reports/experiment_summary.md")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        row = {
            "run": path.stem,
            "dataset": config.get("dataset_name", "cifar100"),
            "setting": config.get("setting", "task-agnostic"),
            "method": data["method"],
            "backbone": data["backbone"],
            "final_acc": data["metrics"]["final_accuracy"],
            "avg_inc_acc": data["metrics"]["average_incremental_accuracy"],
            "forgetting": data["metrics"]["average_forgetting"],
            "bwt": data["metrics"]["backward_transfer"],
            "wall_s": data["timing"]["wall_seconds"],
            "train_s": data["timing"]["method_train_seconds"],
            "predict_s": data["timing"]["method_predict_seconds"],
            "peak_rss_mb": data["resource"]["peak_rss_mb"],
            "proj_density": data["resource"]["projection_density"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out.write_text("# Experiment Summary\n\nNo result JSON files found.\n", encoding="utf-8")
        return

    shown = df.sort_values(["method", "run"]).copy()
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    table = "\n".join(lines)
    text = [
        "# Experiment Summary",
        "",
        "All numbers are produced by the local scripts in this repository. Accuracy is top-1 percentage on the classes visible under each experiment's continual-learning setting.",
        "",
        table,
        "",
    ]
    out.write_text("\n".join(text), encoding="utf-8")
    df.to_csv(out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
