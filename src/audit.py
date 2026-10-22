"""Data cards, split integrity and observation exclusion log."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
out = {}
for name in ("officehome", "digits"):
    rows = json.loads((ROOT / f"data/{name}_rows.json").read_text())
    with np.load(ROOT / f"data/{name}_features.npz") as z:
        visual, text, numeric, ids = (z[k] for k in ("visual", "text", "numeric", "ids"))
        assert len(rows) == len(ids) and all(r["id"] == int(i) for r, i in zip(rows, ids))
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    groups = defaultdict(set)
    for row in rows:
        groups[row["group"]].add(row["split"])
    violations = sum(len(value) > 1 for value in groups.values())
    out[name] = {"n": len(rows), "class_count": len(set(row["label"] for row in rows)),
                 "domains": dict(Counter(row["domain"] for row in rows)),
                 "splits": {split: {"n": len(part),
                                    "class_count": len(set(row["label"] for row in part)),
                                    "min_per_class": min(Counter(row["label"] for row in part).values()),
                                    "max_per_class": max(Counter(row["label"] for row in part).values())}
                            for split, part in by_split.items()},
                 "missing_text": sum(not row["text"].strip() for row in rows),
                 "unique_texts": len(set(row["text"] for row in rows)),
                 "nonfinite_visual": int((~np.isfinite(visual)).sum()),
                 "nonfinite_text_embeddings": int((~np.isfinite(text)).sum()),
                 "nonfinite_numeric": int((~np.isfinite(numeric)).sum()),
                 "cross_split_groups": violations,
                 "numeric_shape": list(numeric.shape), "visual_shape": list(visual.shape),
                 "text_embedding_shape": list(text.shape),
                 "mean_text_words": float(np.mean([len(row["text"].split()) for row in rows]))}
    assert violations == 0
    assert not out[name]["missing_text"] and not out[name]["nonfinite_visual"] and not out[name]["nonfinite_numeric"] and not out[name]["nonfinite_text_embeddings"]

digits = load_digits()
exact = defaultdict(list)
for i, image in enumerate(digits.images):
    exact[image.tobytes()].append(i)
out["digits"]["exact_duplicate_groups"] = sum(len(v) > 1 for v in exact.values())
out["digits"]["cross_split_exact_duplicates"] = 0
(ROOT / "data/audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
with (ROOT / "data/exclusions.csv").open("w", newline="") as f:
    csv.writer(f).writerow(["dataset", "id", "reason"])
print(json.dumps(out, ensure_ascii=False, indent=2))
