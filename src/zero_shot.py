"""Fixed class-prompt SigLIP 2 zero-shot reference, without any fitted head."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from train import CONFIG, ROOT, RESULTS, metrics


def main():
    out = []
    for dataset in CONFIG["datasets"]:
        rows = json.loads((ROOT / f"data/{dataset}_rows.json").read_text())
        meta = json.loads((ROOT / f"data/{dataset}_extraction.json").read_text())
        with np.load(ROOT / f"data/{dataset}_features.npz") as features:
            images = features["visual"]
            prompts = features["prototypes"]
            ids = features["ids"]
        assert all(int(i) == r["id"] for i, r in zip(ids, rows))
        probs = softmax(float(meta["logit_scale"]) * images @ prompts.T, axis=1)
        for split in sorted(set(r["split"] for r in rows)):
            ix = np.array([i for i, r in enumerate(rows) if r["split"] == split])
            truth = np.array([rows[i]["label"] for i in ix], dtype=int)
            np.savez_compressed(RESULTS / f"{dataset}_zero_shot_{split}.npz",
                                probs=probs[ix], truth=truth, ids=ids[ix])
            out.append({"dataset": dataset, "scheme": "zero_shot", "seed": "pretrained",
                        "split": split, **metrics(probs[ix], truth, prompts.shape[0])})
    with (RESULTS / "zero_shot_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
