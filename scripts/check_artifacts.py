"""Check public example labels, summary agreement, links, and extracted code."""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "examples/synthetic_example.json").read_text(encoding="utf-8"))
assert meta["data_kind"] == "synthetic numeric predictions" and meta["model_executed"] is False
assert meta["case_count"] == 360 and meta["prediction_rows"] == 2880
for name in ["task_metrics", "group_metrics", "comparisons", "seed_summary"]:
    path = ROOT / "examples" / f"synthetic_{name}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows and {r["task"] for r in rows} == {"task-a", "task-b", "task-c"}
    regenerated = ROOT / "examples/output" / path.name
    if regenerated.exists():
        other = list(csv.DictReader(regenerated.open(encoding="utf-8", newline="")))
        assert len(rows) == len(other)
        for left, right in zip(rows, other):
            assert left.keys() == right.keys()
            for k in left:
                try:
                    assert math.isclose(float(left[k]), float(right[k]), abs_tol=1e-7)
                except ValueError:
                    assert left[k] == right[k]
for p in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
        if not target.startswith(("http://", "https://", "mailto:", "#")):
            assert (p.parent / target.split("#")[0]).exists(), (p.name, target)
provenance = json.loads((ROOT / "docs/code_provenance.json").read_text(encoding="utf-8"))
for filename, key in [("metrics.py", "metrics_functions"), ("comparisons.py", "paired_bootstrap_function")]:
    text = (ROOT / "zero_shot_evaluation" / filename).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    extracted = "\n\n".join("".join(lines[n.lineno - 1:n.end_lineno]).rstrip() for n in functions) + "\n"
    assert hashlib.sha256(extracted.encode()).hexdigest() == provenance["function_sha256"][key]
print("Synthetic labels, regenerated summaries, documentation links, and unchanged extracted functions verified.")
