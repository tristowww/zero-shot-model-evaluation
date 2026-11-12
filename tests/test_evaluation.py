import copy
import unittest

from zero_shot_evaluation.analysis import summarize
from zero_shot_evaluation.comparisons import paired_bootstrap_accuracy_difference
from zero_shot_evaluation.metrics import confusion_metrics, wilson_interval
from zero_shot_evaluation.synthetic import make_predictions


class PairedComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_exact_zero_difference_and_interval(self):
        same = [True, False, True, True]
        result = paired_bootstrap_accuracy_difference(["0", "0", "1", "1"], same, same, seed=42, replications=100)
        self.assertEqual((result["accuracy_difference"], result["ci_low"], result["ci_high"]), (0.0, 0.0, 0.0))

    def test_pairwise_wins_preserve_known_difference(self):
        result = paired_bootstrap_accuracy_difference(["0", "0", "1", "1", "1"],
                                                      [True, True, True, False, True],
                                                      [False, True, False, False, True], seed=42, replications=100)
        self.assertEqual(result["accuracy_difference"], 0.4)
        self.assertEqual(result["candidate_only_correct"], 2)
        self.assertEqual(result["baseline_only_correct"], 0)

    def test_strata_keep_observed_class_weights(self):
        result = paired_bootstrap_accuracy_difference(["0"] * 80 + ["1"] * 20,
                                                      [True] * 100, [True] * 80 + [False] * 20,
                                                      seed=42, replications=100)
        self.assertEqual((result["accuracy_difference"], result["ci_low"], result["ci_high"]), (0.2, 0.2, 0.2))

    def test_unaligned_inputs_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_accuracy_difference(["0"], [True, False], [True], seed=42)

    def test_empty_and_invalid_resampling_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_accuracy_difference([], [], [], seed=42)
        with self.assertRaises(ValueError):
            paired_bootstrap_accuracy_difference(["0"], [True], [True], seed=42, replications=1)


class MetricsTests(unittest.TestCase):
    def test_known_confusion_table(self):
        rows = [{"label": a, "prediction": b} for a, b in [("1", "1"), ("1", "0"), ("0", "0"), ("0", "1")]]
        result = confusion_metrics(rows, "1")
        self.assertEqual([result[k] for k in ["tp", "tn", "fp", "fn"]], [1, 1, 1, 1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_absent_class_does_not_invent_balanced_accuracy(self):
        result = confusion_metrics([{"label": "1", "prediction": "1"}], "1")
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(wilson_interval(0, 0), (None, None))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = make_predictions(cases_per_task=8)

    def test_complete_fixture_is_reproducible_and_keeps_all_generated_sets(self):
        first = summarize(self.rows, replications=100)
        second = summarize(self.rows, replications=100)
        self.assertEqual(first, second)
        self.assertEqual(first["case_count"], 24)
        self.assertEqual(first["prediction_rows"], 192)
        self.assertEqual(len(first["comparisons"]), 21)
        self.assertTrue(all(s["prompt_sets"] == 5 for s in first["seed_summary"]))

    def test_duplicate_case_condition_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(self.rows + [self.rows[0]], replications=100)

    def test_incomplete_pairing_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete condition"):
            summarize(self.rows[1:], replications=100)

    def test_inconsistent_labels_or_groups_rejected(self):
        for field in ["label", "group"]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = "changed"
            with self.assertRaisesRegex(ValueError, "metadata"):
                summarize(rows, replications=100)

    def test_unknown_prediction_label_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["prediction"] = "unknown"
        with self.assertRaisesRegex(ValueError, "observed labels"):
            summarize(rows, replications=100)

    def test_missing_baseline_or_empty_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            summarize([r for r in self.rows if r["condition"] != "single"], replications=100)
        with self.assertRaises(ValueError):
            summarize([], replications=100)

    def test_blank_and_null_metadata_rejected(self):
        for value in [None, "", "   "]:
            rows = copy.deepcopy(self.rows)
            rows[0]["case_id"] = value
            with self.assertRaisesRegex(ValueError, "nonempty"):
                summarize(rows, replications=100)


if __name__ == "__main__":
    unittest.main()
