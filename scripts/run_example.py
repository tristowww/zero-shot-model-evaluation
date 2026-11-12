"""Generate the portable example, without downloading anything."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zero_shot_evaluation.analysis import summarize
from zero_shot_evaluation.synthetic import make_predictions


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.8f}" if isinstance(v, float) else v for k, v in row.items()})


def build(output: Path):
    result = summarize(make_predictions(), replications=2000)
    output.mkdir(parents=True, exist_ok=True)
    for key in ["task_metrics", "group_metrics", "comparisons", "seed_summary"]:
        write_csv(output / f"synthetic_{key}.csv", result[key])
    metadata = {"data_kind": "synthetic numeric predictions", "model_executed": False,
                "seed": 20261112, "bootstrap_seed": 1701, "bootstrap_replications": 2000,
                "cases_per_task": 120, "case_count": result["case_count"],
                "prediction_rows": result["prediction_rows"], "conditions": result["conditions"],
                "warning": "Invented outcomes for demonstrating the method. These are not CLIP, BLOOM, or Qwen research results."}
    (output / "synthetic_example.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples/output")
    build(parser.parse_args().output_dir)
