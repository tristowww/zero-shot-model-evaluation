"""Validate aligned categorical predictions before comparing conditions."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .comparisons import paired_bootstrap_accuracy_difference
from .metrics import confusion_metrics


def summarize(rows: list[dict], *, seed: int = 1701, replications: int = 2000) -> dict:
    """Summarize complete paired cases; never select a best generated set."""
    if not rows:
        raise ValueError("Predictions must contain at least one row")
    required = {"case_id", "task", "label", "prediction", "condition", "group"}
    by_case = {}
    conditions = set()
    labels_by_task = defaultdict(set)
    for row in rows:
        if not required.issubset(row) or any(not isinstance(row[k], str) or not row[k].strip() for k in required):
            raise ValueError("Every prediction needs nonempty case, task, label, condition, and group fields")
        key = (row["task"], row["case_id"])
        case = by_case.setdefault(key, {"label": row["label"], "group": row["group"], "predictions": {}})
        if (case["label"], case["group"]) != (row["label"], row["group"]):
            raise ValueError("Labels and group metadata must agree across conditions")
        condition = row["condition"]
        if condition in case["predictions"]:
            raise ValueError("Duplicate condition for the same task and case")
        case["predictions"][condition] = row["prediction"]
        conditions.add(condition)
        labels_by_task[row["task"]].add(row["label"])
    if "single" not in conditions:
        raise ValueError("The single baseline is required")
    for (task, _), case in by_case.items():
        if set(case["predictions"]) != conditions:
            raise ValueError("Every case must have the same complete condition set")
        if len(labels_by_task[task]) != 2:
            raise ValueError("This companion expects two observed labels in each task")
        if any(p not in labels_by_task[task] for p in case["predictions"].values()):
            raise ValueError("Predictions must use the task's observed labels")
    task_rows, contrasts, groups, seeds = [], [], [], []
    for task in sorted(labels_by_task):
        cases = [v for (t, _), v in sorted(by_case.items()) if t == task]
        labels = [c["label"] for c in cases]
        positive = sorted(labels_by_task[task])[1]
        baseline = [c["predictions"]["single"] == c["label"] for c in cases]
        for condition in sorted(conditions):
            prediction_rows = [{"label": c["label"], "prediction": c["predictions"][condition]} for c in cases]
            task_rows.append({"task": task, "condition": condition, **confusion_metrics(prediction_rows, positive)})
            for group in sorted({c["group"] for c in cases}):
                subset = [{"label": c["label"], "prediction": c["predictions"][condition]} for c in cases if c["group"] == group]
                groups.append({"task": task, "condition": condition, "group": group, **confusion_metrics(subset, positive)})
            if condition != "single":
                candidate = [c["predictions"][condition] == c["label"] for c in cases]
                result = paired_bootstrap_accuracy_difference(labels, candidate, baseline, seed=seed, replications=replications)
                contrasts.append({"task": task, "condition": condition, "baseline": "single", **result})
        generated = [r for r in contrasts if r["task"] == task and r["condition"].startswith("generated-")]
        if generated:
            values = [r["accuracy_difference"] for r in generated]
            seeds.append({"task": task, "prompt_sets": len(values), "difference_mean": statistics.mean(values),
                          "difference_sd": statistics.stdev(values) if len(values) > 1 else None,
                          "difference_min": min(values), "difference_max": max(values),
                          "sets_improved": sum(v > 0 for v in values), "sets_tied": sum(v == 0 for v in values),
                          "sets_worse": sum(v < 0 for v in values)})
    for row in contrasts:
        if not all(math.isfinite(row[k]) for k in ["accuracy_difference", "ci_low", "ci_high"]):
            raise ValueError("Nonfinite comparison")
    return {"case_count": len(by_case), "prediction_rows": len(rows), "conditions": sorted(conditions),
            "task_metrics": task_rows, "group_metrics": groups, "comparisons": contrasts, "seed_summary": seeds}
