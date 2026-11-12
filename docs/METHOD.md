# Reading the paired comparisons

When two conditions evaluate the same cases, their results are paired. A comparison should preserve that pairing, including which cases both conditions get right or wrong.

The extracted bootstrap function assigns each case a correctness difference of -1, 0, or +1. It resamples the counts within each observed task-label stratum and weights each stratum by its original size. For this accuracy contrast, that is equivalent to resampling intact case pairs within labels. It reports a percentile 95% interval for the candidate-minus-baseline accuracy difference.

## Input checks

The public wrapper expects binary categorical predictions with `case_id`, `task`, `label`, `prediction`, `condition`, and `group`. Within each task, a case must have one row for every condition and consistent label/group metadata. Unknown predicted labels, duplicate rows, and incomplete condition sets fail before metrics are calculated.

The positive label for the companion's binary confusion metrics is the lexicographically later observed label. In the numeric example, this is `1`. Set an explicit positive-label policy before adapting these helpers to substantive labels. Accuracy differences do not depend on that convention.

The wrapper also reports descriptive accuracy and confusion metrics for each supplied group. It does not calculate group-level paired intervals or validate what a group label means. The synthetic groups are invented slices, not demographic groups.

## Two different kinds of uncertainty

The bootstrap interval describes case-sampling uncertainty for one fixed candidate set and baseline. The five generated sets are reported individually, with their mean, standard deviation, range, and directional count. Their variation is not included in each case-bootstrap interval.

These intervals assume cases are suitable independent sampling units within task and label. Repeated people, clustered observations, dependent cases, or other sampling designs require an appropriate alternative. The example is a demonstration, not evidence of calibrated coverage for a real population.

## Synthetic example

The generator creates three tasks with 120 cases each and eight conditions. Fixed, deliberately different correctness patterns illustrate a favorable comparison, mixed generated-set results, and an unfavorable comparison. All conditions on a case share an invented difficulty draw so pairing is visible.

No prompts, images, model predictions, or participant records are used. The generation seed is 20261112. The demonstration uses 2,000 bootstrap draws with seed 1701; the original research used 10,000 draws. Run `python scripts/run_example.py` from the repository root to regenerate the CSV summaries and example metadata.

`scripts/plot_example.py` reads the committed synthetic comparisons. It plots every generated set against the same single baseline, uses common axes across tasks, and labels the figure as synthetic. It does not draw the original research findings.
