"""Retrain selected fusion family without one input at a time.

These are post-selection diagnostic ablations, not new confirmatory candidates.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch
from train import CONFIG, RESULTS, load_data, metrics, probabilities, train_one


def main():
    selection = json.loads((RESULTS / "selection.json").read_text())
    all_rows = []
    for dataset in CONFIG["datasets"]:
        data = load_data(dataset)
        original = next(s for s in CONFIG["schemes"] if s["key"] ==
                        selection[dataset]["fusion_scheme"])
        for absent in ["visual", "text", "numeric"]:
            scheme = {**original, "modalities": [m for m in original["modalities"] if m != absent]}
            for seed in CONFIG["train_seeds"]:
                model, val, epoch, seconds, history = train_one(data, scheme, seed)
                label = f"{dataset}_retrain_without_{absent}_{seed}"
                torch.save(model.state_dict(), RESULTS / f"{label}.pt")
                (RESULTS / f"{label}_history.json").write_text(json.dumps(history))
                for split in ["validation", "test"]:
                    ix = data["ix"][split]
                    probs, ms = probabilities(model, data["x"], ix)
                    truth = np.array([data["rows"][i]["label"] for i in ix], dtype=int)
                    ids = np.array([data["rows"][i]["id"] for i in ix], dtype=int)
                    np.savez_compressed(RESULTS / f"{label}_{split}.npz",
                                        probs=probs, truth=truth, ids=ids)
                    all_rows.append({"dataset": dataset, "ablation": f"without_{absent}",
                                     "seed": seed, "split": split, **metrics(probs, truth, data["classes"]),
                                     "best_source_validation_macro_f1": val,
                                     "best_epoch": epoch, "training_seconds": seconds,
                                     "head_ms_per_image": ms})
    with (RESULTS / "retrained_ablations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summaries = []
    for dataset in CONFIG["datasets"]:
        for absent in ["visual", "text", "numeric"]:
            arrays = [np.load(RESULTS / f"{dataset}_retrain_without_{absent}_{seed}_test.npz")
                      for seed in CONFIG["train_seeds"]]
            y = arrays[0]["truth"]
            p = np.mean([a["probs"] for a in arrays], axis=0)
            score = metrics(p, y, 65 if dataset == "officehome" else 10)
            summaries.append({"dataset": dataset, "ablation": f"without_{absent}", **score})
            print(dataset, "retrained without", absent, score["macro_f1"], flush=True)
    with (RESULTS / "retrained_ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
