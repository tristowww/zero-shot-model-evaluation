"""Metrics extracted unchanged from the research implementation."""
from __future__ import annotations

import math
from typing import Iterable


def safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    if total == 0:
        return (None, None)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return (centre - margin, centre + margin)

def confusion_metrics(rows: Iterable[dict], positive_label: str) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        actual_positive = row["label"] == positive_label
        predicted_positive = row["prediction"] == positive_label
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1
    n = tp + tn + fp + fn
    correct = tp + tn
    accuracy = safe_divide(correct, n)
    tpr = safe_divide(tp, tp + fn)
    tnr = safe_divide(tn, tn + fp)
    balanced = None if tpr is None or tnr is None else (tpr + tnr) / 2
    ci_low, ci_high = wilson_interval(correct, n)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "balanced_accuracy": balanced,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": None if tnr is None else 1 - tnr,
        "fnr": None if tpr is None else 1 - tpr,
    }

def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
