"""Predefined matched comparisons that separate fusion from alignment."""
from __future__ import annotations

import csv

from analyze import paired_interval, paired_permutation, read_predictions
from train import CONFIG, RESULTS, metrics


def main():
    rows = []
    pairs = [
        ("visual_mmd", "visual", "effect_of_mmd_visual"),
        ("visual_coral", "visual", "effect_of_coral_visual"),
        ("mean_mmd", "visual_mmd", "fusion_effect_given_mmd"),
        ("mean_coral", "visual_coral", "fusion_effect_given_coral"),
        ("zero_shot", "visual", "zero_shot_vs_trained_visual"),
    ]
    for dataset in CONFIG["datasets"]:
        classes = 65 if dataset == "officehome" else 10
        for a_name, b_name, label in pairs:
            ids, y, a = read_predictions(dataset, a_name, "test")
            b_ids, b_y, b = read_predictions(dataset, b_name, "test")
            assert (ids == b_ids).all() and (y == b_y).all()
            delta, (low, high) = paired_interval(a, b, y, classes,
                                                 CONFIG["bootstrap_repetitions"], 310)
            p = paired_permutation(a, b, y, classes,
                                   CONFIG["permutation_repetitions"], 410)
            rows.append({"dataset": dataset, "comparison": label, "method": a_name,
                         "control": b_name, "method_macro_f1": metrics(a, y, classes)["macro_f1"],
                         "control_macro_f1": metrics(b, y, classes)["macro_f1"],
                         "difference": delta, "ci_low": low, "ci_high": high,
                         "paired_permutation_two_sided_p": p})
    with (RESULTS / "paired_controls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
