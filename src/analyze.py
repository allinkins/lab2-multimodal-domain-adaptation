"""Locked selection, paired uncertainty, missing-modality ablations, errors."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train import CONFIG, Net, ROOT, RESULTS, load_data, metrics, probabilities


SEEDS = CONFIG["train_seeds"]


def read_predictions(name: str, scheme: str, split: str):
    if scheme == "zero_shot":
        with np.load(RESULTS / f"{name}_zero_shot_{split}.npz") as a:
            return a["ids"].copy(), a["truth"].copy(), a["probs"].copy()
    arrays = [np.load(RESULTS / f"{name}_{scheme}_{seed}_{split}.npz") for seed in SEEDS]
    ids, truth = arrays[0]["ids"], arrays[0]["truth"]
    assert all(np.array_equal(ids, a["ids"]) and np.array_equal(truth, a["truth"]) for a in arrays)
    return ids, truth, np.mean([a["probs"] for a in arrays], axis=0)


def paired_interval(a: np.ndarray, b: np.ndarray, y: np.ndarray, classes: int,
                    repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in range(classes)]
    assert all(len(group) for group in groups)
    observed = metrics(a, y, classes)["macro_f1"] - metrics(b, y, classes)["macro_f1"]
    sampled = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        sampled.append(metrics(a[indices], y[indices], classes)["macro_f1"] -
                       metrics(b[indices], y[indices], classes)["macro_f1"])
    return observed, list(np.quantile(sampled, [0.025, 0.975]))


def paired_permutation(a: np.ndarray, b: np.ndarray, y: np.ndarray, classes: int,
                       repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    observed = metrics(a, y, classes)["macro_f1"] - metrics(b, y, classes)["macro_f1"]
    extreme = 0
    for _ in range(repetitions):
        swap = rng.random(len(y)) < 0.5
        ap, bp = a.copy(), b.copy()
        ap[swap], bp[swap] = b[swap], a[swap]
        diff = metrics(ap, y, classes)["macro_f1"] - metrics(bp, y, classes)["macro_f1"]
        extreme += abs(diff) >= abs(observed) - 1e-12
    return (extreme + 1) / (repetitions + 1)


def ablate(name: str, scheme: str, data: dict, split: str):
    active = {"visual", "text", "numeric"}
    options = {"full": active, "without_visual": active - {"visual"},
               "without_text": active - {"text"}, "without_numeric": active - {"numeric"},
               "visual_only_at_inference": {"visual"}}
    fusion_config = next(s for s in CONFIG["schemes"] if s["key"] == scheme)
    indices = data["ix"][split]
    truth = np.array([data["rows"][i]["label"] for i in indices], dtype=int)
    out = []
    for label, mask in options.items():
        scores = []
        latency = []
        for seed in SEEDS:
            model = Net(data["dimensions"], data["classes"], fusion_config["modalities"],
                        fusion_config["fusion"])
            state = torch.load(RESULTS / f"{name}_{scheme}_{seed}.pt", weights_only=True,
                               map_location="cpu")
            model.load_state_dict(state)
            p, ms = probabilities(model, data["x"], indices, mask)
            scores.append(p)
            latency.append(ms)
        score = metrics(np.mean(scores, axis=0), truth, data["classes"])
        out.append({"dataset": name, "selected_scheme": scheme, "split": split,
                    "ablation": label, **score, "head_ms_per_image": float(np.mean(latency))})
    return out


def main():
    frame = pd.read_csv(RESULTS / "metrics.csv")
    selection = {}
    comparisons = []
    ablations = []
    summaries = []
    errors = []
    for name in CONFIG["datasets"]:
        val = frame[(frame.dataset == name) & (frame.split == "validation")]
        by_scheme = val.groupby("scheme")["macro_f1"].mean()
        fusion_candidates = [s["key"] for s in CONFIG["schemes"] if len(s["modalities"]) > 1]
        unimodal_candidates = ["visual", "text", "numeric", "zero_shot"]
        zs = pd.read_csv(RESULTS / "zero_shot_metrics.csv")
        by_scheme["zero_shot"] = float(zs[(zs.dataset == name) &
                                           (zs.split == "validation")].macro_f1.iloc[0])
        chosen = max(fusion_candidates, key=lambda s: by_scheme[s])
        baseline = max(unimodal_candidates, key=lambda s: by_scheme[s])
        selection[name] = {"fusion_scheme": chosen, "unimodal_scheme": baseline,
                           "source_validation_mean_macro_f1": {k: float(v) for k, v in by_scheme.items()},
                           "rule": CONFIG["model_selection_rule"],
                           "test_consulted_for_current_model_choice": False,
                           "evaluation_status": CONFIG["evaluation_status"]}
        data = load_data(name)
        classes = data["classes"]
        for split in ["test"] + (["new_slice", "external"] if name == "officehome" else []):
            ids, y, p = read_predictions(name, chosen, split)
            ids_base, y_base, q = read_predictions(name, baseline, split)
            assert np.array_equal(ids, ids_base) and np.array_equal(y, y_base)
            effect, interval = paired_interval(p, q, y, classes, CONFIG["bootstrap_repetitions"], 14)
            pvalue = paired_permutation(p, q, y, classes,
                                        CONFIG["permutation_repetitions"], 25)
            comparisons.append({"dataset": name, "split": split, "fusion": chosen,
                                "unimodal": baseline, "fusion_macro_f1": metrics(p, y, classes)["macro_f1"],
                                "unimodal_macro_f1": metrics(q, y, classes)["macro_f1"],
                                "difference": effect, "ci_low": interval[0], "ci_high": interval[1],
                                "paired_permutation_two_sided_p": pvalue,
                                "bootstrap_repetitions": CONFIG["bootstrap_repetitions"],
                                "permutation_repetitions": CONFIG["permutation_repetitions"]})
            summaries.extend([{"dataset": name, "split": split, "scheme": scheme,
                               **metrics(read_predictions(name, scheme, split)[2], y, classes)}
                              for scheme in [s["key"] for s in CONFIG["schemes"]] + ["zero_shot"]])
            if split == "test":
                for row in ablate(name, chosen, data, split):
                    ablations.append(row)
                pred, base_pred = p.argmax(1), q.argmax(1)
                lookup = {r["id"]: r for r in data["rows"]}
                for i, item_id in enumerate(ids):
                    if pred[i] == y[i]:
                        continue
                    row = lookup[int(item_id)]
                    errors.append({"dataset": name, "id": int(item_id), "true_label": int(y[i]),
                                   "fusion_prediction": int(pred[i]),
                                   "baseline_prediction": int(base_pred[i]),
                                   "baseline_correct": bool(base_pred[i] == y[i]),
                                   "fusion_confidence": float(p[i].max()),
                                   "text": row["text"]})
        print("SELECT", name, chosen, "vs", baseline, "target effect", comparisons[-(3 if name == "officehome" else 1)]["difference"], flush=True)
    (RESULTS / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    for filename, rows in [("paired_comparisons.csv", comparisons),
                           ("ensemble_summary.csv", summaries),
                           ("missing_modality.csv", ablations),
                           ("test_errors.csv", errors)]:
        with (RESULTS / filename).open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
