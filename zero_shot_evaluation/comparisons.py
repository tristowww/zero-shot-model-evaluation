"""Paired bootstrap extracted unchanged from the research implementation."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def paired_bootstrap_accuracy_difference(
    labels: Sequence[str],
    candidate_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
    *,
    seed: int,
    replications: int = 10_000,
) -> dict:
    """Estimate a paired accuracy difference with a label-stratified bootstrap.

    Each observation is resampled as an intact candidate/baseline pair. The
    implementation samples counts of {-1, 0, +1} differences, which is exactly
    equivalent to resampling individual rows but is substantially faster.
    """
    if not (len(labels) == len(candidate_correct) == len(baseline_correct)):
        raise ValueError("Paired inputs must have the same length")
    if not labels:
        raise ValueError("Paired bootstrap requires at least one observation")
    if replications < 2:
        raise ValueError("replications must be at least 2")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Publication comparisons require NumPy; install the 'analysis' extra"
        ) from exc

    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    observed_sum = 0
    wins = ties = losses = 0
    for label, candidate, baseline in zip(
        labels, candidate_correct, baseline_correct
    ):
        difference = int(candidate) - int(baseline)
        strata[label][difference + 1] += 1
        observed_sum += difference
        if difference > 0:
            wins += 1
        elif difference < 0:
            losses += 1
        else:
            ties += 1

    rng = np.random.default_rng(seed)
    bootstrap_sums = np.zeros(replications, dtype=np.int64)
    for counts in strata.values():
        stratum_n = sum(counts)
        draws = rng.multinomial(
            stratum_n,
            np.asarray(counts, dtype=float) / stratum_n,
            size=replications,
        )
        bootstrap_sums += draws[:, 2] - draws[:, 0]

    estimates = bootstrap_sums / len(labels)
    return {
        "n": len(labels),
        "candidate_accuracy": sum(candidate_correct) / len(labels),
        "baseline_accuracy": sum(baseline_correct) / len(labels),
        "accuracy_difference": observed_sum / len(labels),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(estimates > 0)),
        "candidate_only_correct": wins,
        "both_same_correctness": ties,
        "baseline_only_correct": losses,
        "replications": replications,
    }
