"""Plot the committed synthetic comparisons; no research-data access."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "examples/synthetic_comparisons.csv").open(encoding="utf-8")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "synthetic-zero-shot-example"})
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), sharex=True, sharey=True)
for ax, task in zip(axes, ["task-a", "task-b", "task-c"]):
    values = [r for r in rows if r["task"] == task and r["condition"].startswith("generated-")]
    for i, r in enumerate(values):
        mean, low, high = [float(r[k]) * 100 for k in ["accuracy_difference", "ci_low", "ci_high"]]
        ax.plot([low, high], [i, i], color="#427768", linewidth=2)
        ax.plot(mean, i, "o", color="#214d40", markersize=6)
    ax.axvline(0, color="#767676", linewidth=1, linestyle="--")
    ax.set_title(task.replace("task-", "Toy task ").upper(), fontsize=12, pad=14)
    ax.set_yticks(range(5), [f"Generated set {i}" for i in range(1, 6)])
    ax.set_xlim(-25, 25)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle("Synthetic example: compare every generated set with the single baseline", fontsize=15, y=0.98)
fig.supxlabel("Accuracy difference (percentage points); bars show paired 95% bootstrap intervals", fontsize=11, y=0.12)
fig.text(0.5, 0.025, "Invented outcomes, 120 cases per toy task. No model was run. These are not the research findings.",
         ha="center", fontsize=10, color="#555555")
fig.tight_layout(rect=(0, 0.16, 1, 0.91))
fig.savefig(ROOT / "examples/synthetic_comparison.png", dpi=160, facecolor="white")
fig.savefig(ROOT / "examples/synthetic_comparison.svg", metadata={"Date": None}, facecolor="white")
svg_path = ROOT / "examples/synthetic_comparison.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
print("Saved synthetic comparison PNG and SVG.")
