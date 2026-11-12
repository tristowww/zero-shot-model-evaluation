"""Invented categorical outcomes; no images, people, or model calls."""
import random


def make_predictions(seed: int = 20261112, cases_per_task: int = 120) -> list[dict]:
    if cases_per_task < 4 or cases_per_task % 4:
        raise ValueError("Use a positive multiple of four, at least four")
    rng = random.Random(seed)
    # Candidate outcomes share a difficulty draw with the baselines. These
    # deliberately different patterns illustrate pairing and seed variability.
    thresholds = {
        "task-a": [0.70, 0.73, 0.76, 0.77, 0.80, 0.83, 0.78, 0.81],
        "task-b": [0.80, 0.78, 0.82, 0.76, 0.82, 0.78, 0.85, 0.74],
        "task-c": [0.85, 0.84, 0.86, 0.72, 0.76, 0.79, 0.74, 0.77],
    }
    conditions = ["single", "human", "historical"] + [f"generated-{i}" for i in range(1, 6)]
    rows = []
    for task, rates in thresholds.items():
        for i in range(cases_per_task):
            label = str(i % 2)
            difficulty = rng.random()
            group = "slice-a" if (i // 2) % 2 else "slice-b"
            for condition, rate in zip(conditions, rates):
                prediction = label if difficulty < rate else str(1 - int(label))
                rows.append({"case_id": f"synthetic-{task}-{i:03d}", "task": task, "label": label,
                             "prediction": prediction, "condition": condition, "group": group})
    return rows
