"""Scheme intervals, paired alignment effects, charts and error strata."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lab2_mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import read_predictions
from train import CONFIG, RESULTS, metrics


def interval(probs, y, classes, reps=2000, seed=20261020):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in range(classes)]
    values = []
    for _ in range(reps):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(metrics(probs[indices], y[indices], classes)["macro_f1"])
    return np.quantile(values, (0.025, 0.975)).tolist()


source = pd.read_csv(RESULTS / "metrics.csv")
rows = []
effects = []
for dataset in CONFIG["datasets"]:
    classes = 65 if dataset == "officehome" else 10
    for scheme in [s["key"] for s in CONFIG["schemes"]] + ["zero_shot"]:
        _, y, p = read_predictions(dataset, scheme, "test")
        scores = metrics(p, y, classes)
        low, high = interval(p, y, classes)
        runs = source[(source.dataset == dataset) & (source.scheme == scheme) & (source.split == "test")]
        rows.append({"dataset": dataset, "scheme": scheme, **scores,
                     "macro_f1_ci_low": low, "macro_f1_ci_high": high,
                     "run_macro_f1_mean": runs.macro_f1.mean() if len(runs) else scores["macro_f1"],
                     "run_macro_f1_sd": runs.macro_f1.std() if len(runs) else 0,
                     "head_ms_per_image_mean": runs.inference_ms_per_image.mean() if len(runs) else 0,
                     "training_seconds_mean": runs.training_seconds.mean() if len(runs) else 0})
    for scope in ["visual", "mean", "concat"]:
        ref = source[(source.dataset == dataset) & (source.scheme == scope) & (source.split == "test")]
        for method in ["mmd", "coral"]:
            cur = source[(source.dataset == dataset) & (source.scheme == f"{scope}_{method}") & (source.split == "test")]
            joined = ref[["seed", "macro_f1"]].merge(cur[["seed", "macro_f1"]], on="seed", suffixes=("_ref", "_method"))
            delta = joined.macro_f1_method - joined.macro_f1_ref
            effects.append({"dataset": dataset, "scope": scope, "alignment": method,
                            "paired_mean_delta": delta.mean(), "paired_sd_delta": delta.std(),
                            "seed_differences": json.dumps(delta.tolist())})
    for method in ["", "_mmd", "_coral"]:
        vis = source[(source.dataset == dataset) & (source.scheme == f"visual{method}") & (source.split == "test")]
        fus = source[(source.dataset == dataset) & (source.scheme == f"mean{method}") & (source.split == "test")]
        joined = vis[["seed", "macro_f1"]].merge(fus[["seed", "macro_f1"]], on="seed", suffixes=("_visual", "_fusion"))
        delta = joined.macro_f1_fusion - joined.macro_f1_visual
        effects.append({"dataset": dataset, "scope": "fusion-minus-visual", "alignment": method.lstrip("_") or "none",
                        "paired_mean_delta": delta.mean(), "paired_sd_delta": delta.std(),
                        "seed_differences": json.dumps(delta.tolist())})

for path, content in [("scheme_intervals.csv", rows), ("effects.csv", effects)]:
    with (RESULTS / path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(content[0]))
        writer.writeheader()
        writer.writerows(content)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
titles = {"officehome": "Office-Home: Product → Clipart", "digits": "Digits: original → inverted"}
names = {"mean": "Mean", "mean_mmd": "Mean MMD", "mean_coral": "Mean CORAL",
         "concat": "Concat", "concat_mmd": "Concat MMD", "concat_coral": "Concat CORAL",
         "gated": "Gated", "gated_coral": "Gated CORAL"}
for ax, dataset in zip(axes, CONFIG["datasets"]):
    part = [r for r in rows if r["dataset"] == dataset and r["scheme"] in names]
    x = np.arange(len(part))
    y = np.array([r["macro_f1"] for r in part])
    colors = ["#405f88" if r["scheme"].startswith("concat") else
              "#b65c50" if r["scheme"].startswith("mean") else "#69876e" for r in part]
    ax.bar(x, y, color=colors)
    ax.set_xticks(x, [names[r["scheme"]] for r in part], rotation=55, ha="right", fontsize=8)
    ax.set_title(titles[dataset])
    ax.set_ylabel("Macro-F1 on target test")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(RESULTS / "target_macro_f1.png", dpi=180, bbox_inches="tight")

errors = pd.read_csv(RESULTS / "test_errors.csv")
strata = []
for dataset, part in errors.groupby("dataset"):
    strata.append({"dataset": dataset, "total_errors_selected_fusion": len(part),
                   "baseline_correct_selected_wrong": int(part.baseline_correct.sum()),
                   "high_confidence_gt_0_8": int((part.fusion_confidence > 0.8).sum()),
                   "top_wrong_predictions": json.dumps(Counter(part.fusion_prediction).most_common(5))})
(RESULTS / "error_strata.json").write_text(json.dumps(strata, ensure_ascii=False, indent=2))
print("Wrote scheme intervals, effects, chart, error strata")
