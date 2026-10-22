"""Frozen SigLIP 2 features: eight fusion schemes and unimodal controls."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/protocol.json").read_text())
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
torch.set_num_threads(4)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if mask.any():
            out += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(out)


def metrics(probs: np.ndarray, y: np.ndarray, classes: int) -> dict:
    pred = probs.argmax(1)
    truth = np.eye(classes)[y]
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=range(classes), average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "ece": ece(probs, y), "multiclass_brier": float(np.mean(np.sum((probs - truth) ** 2, axis=1))),
            "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean())}


def load_data(name: str) -> dict:
    rows = json.loads((ROOT / f"data/{name}_rows.json").read_text())
    with np.load(ROOT / f"data/{name}_features.npz") as data:
        visual, text, numeric, ids = (data[k].copy() for k in ("visual", "text", "numeric", "ids"))
    assert all(int(id_) == row["id"] for id_, row in zip(ids, rows))
    ix = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s], dtype=np.int64)
          for s in sorted({r["split"] for r in rows})}
    assert not set(ix["source_train"]) & set(ix["target_adapt"])
    inputs = {"visual": visual.astype(np.float32), "text": text.astype(np.float32),
              "numeric": numeric.astype(np.float32)}
    scaling = {}
    for kind, matrix in inputs.items():
        mean = matrix[ix["source_train"]].mean(0)
        std = np.maximum(matrix[ix["source_train"]].std(0), 1e-3)
        inputs[kind] = np.clip((matrix - mean) / std, -6, 6).astype(np.float32)
        scaling[kind] = {"mean": mean.tolist(), "std": std.tolist()}
        assert np.isfinite(inputs[kind]).all()
    (RESULTS / f"{name}_scaling.json").write_text(json.dumps(scaling))
    return {"rows": rows, "ix": ix, "x": {k: torch.from_numpy(v) for k, v in inputs.items()},
            "classes": 65 if name == "officehome" else 10,
            "dimensions": {k: v.shape[1] for k, v in inputs.items()}}


class Net(torch.nn.Module):
    def __init__(self, dimensions: dict, classes: int, active: list[str], fusion: str = "mean"):
        super().__init__()
        self.active = tuple(active)
        self.fusion = fusion
        if fusion not in {"mean", "concat", "gated"}:
            raise ValueError(f"Unknown fusion {fusion}")
        hidden = CONFIG["hidden_dim"]
        self.branches = torch.nn.ModuleDict({k: torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU()) for k, d in dimensions.items()})
        if fusion == "gated":
            self.gates = torch.nn.Parameter(torch.zeros(len(self.active)))
        mapper_input = hidden * len(self.active) if fusion == "concat" else hidden
        self.mapper = torch.nn.Sequential(torch.nn.Linear(mapper_input, hidden), torch.nn.ReLU())
        self.classifier = torch.nn.Linear(hidden, classes)

    def forward(self, x: dict, mask: set[str] | None = None):
        modalities = self.active if mask is None else tuple(k for k in self.active if k in mask)
        if not modalities:
            raise ValueError("At least one modality must remain")
        if self.fusion == "concat":
            # Keep trained slot positions fixed. A missing stream receives zeros.
            encoded = [self.branches[k](x[k]) if k in modalities else
                       torch.zeros((len(x[k]), CONFIG["hidden_dim"]), dtype=x[k].dtype, device=x[k].device)
                       for k in self.active]
            h = torch.cat(encoded, dim=1)
        elif self.fusion == "gated":
            chosen = [i for i, k in enumerate(self.active) if k in modalities]
            weights = torch.softmax(self.gates[chosen], dim=0)
            h = sum(weights[j] * self.branches[self.active[i]](x[self.active[i]])
                    for j, i in enumerate(chosen))
        else:
            h = sum(self.branches[k](x[k]) for k in modalities) / len(modalities)
        h = self.mapper(h)
        return h, self.classifier(h)


def align(source: torch.Tensor, target: torch.Tensor, method: str) -> torch.Tensor:
    if method == "none":
        return source.sum() * 0
    if method == "coral":
        a, b = source - source.mean(0), target - target.mean(0)
        ca, cb = a.T @ a / (len(a) - 1), b.T @ b / (len(b) - 1)
        return ((ca - cb) ** 2).mean() / 4
    z = torch.cat([source, target])
    sq = (z * z).sum(1, keepdim=True)
    dist = (sq + sq.T - 2 * z @ z.T).clamp_min(0) / z.shape[1]
    kernel = sum(torch.exp(-dist / (2 * s * s)) for s in (0.5, 1, 2)) / 3
    n = len(source)
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()


def probabilities(model: Net, x: dict, indices: np.ndarray,
                  mask: set[str] | None = None) -> tuple[np.ndarray, float]:
    model.eval()
    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            sub = indices[start:start + 128]
            _, logits = model({k: v[sub] for k, v in x.items()}, mask)
            batches.append(torch.softmax(logits, dim=1).numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(batches), elapsed / len(indices) * 1000


def train_one(data: dict, scheme: dict, seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ix, x, rows = data["ix"], data["x"], data["rows"]
    y_source = torch.tensor([rows[i]["label"] for i in ix["source_train"]], dtype=torch.long)
    y_val = np.array([rows[i]["label"] for i in ix["validation"]], dtype=np.int64)
    model = Net(data["dimensions"], data["classes"], scheme["modalities"],
                scheme.get("fusion", "mean"))
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                            weight_decay=CONFIG["weight_decay"])
    best, best_epoch, best_state = -1.0, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        perm = rng.permutation(len(ix["source_train"]))
        losses = []
        for start in range(0, len(perm), CONFIG["batch_size"]):
            local = perm[start:start + CONFIG["batch_size"]]
            source_indices = ix["source_train"][local]
            src_h, src_logits = model({k: v[source_indices] for k, v in x.items()})
            ce = torch.nn.functional.cross_entropy(src_logits, y_source[local])
            if scheme["alignment"] != "none":
                target_indices = rng.choice(ix["target_adapt"], size=len(source_indices),
                                            replace=len(ix["target_adapt"]) < len(source_indices))
                target_h, _ = model({k: v[target_indices] for k, v in x.items()})
                loss = ce + CONFIG["alignment_weight"] * align(src_h, target_h, scheme["alignment"])
            else:
                loss = ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val_probs, _ = probabilities(model, x, ix["validation"])
        val = metrics(val_probs, y_val, data["classes"])["macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "source_validation_macro_f1": val})
        if val > best:
            best, best_epoch = val, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, best, best_epoch, elapsed, history


def main():
    all_rows = []
    for name in CONFIG["datasets"]:
        data = load_data(name)
        print("DATASET", name, {k: len(v) for k, v in data["ix"].items()}, data["dimensions"], flush=True)
        for scheme in CONFIG["schemes"]:
            for seed in CONFIG["train_seeds"]:
                model, best, best_epoch, elapsed, history = train_one(data, scheme, seed)
                label = f"{name}_{scheme['key']}_{seed}"
                torch.save(model.state_dict(), RESULTS / f"{label}.pt")
                (RESULTS / f"{label}_history.json").write_text(json.dumps(history))
                for split in ["validation", "test"] + (["new_slice", "external"] if name == "officehome" else []):
                    indices = data["ix"][split]
                    probs, latency = probabilities(model, data["x"], indices)
                    truth = np.array([data["rows"][i]["label"] for i in indices], dtype=np.int64)
                    ids = np.array([data["rows"][i]["id"] for i in indices], dtype=np.int64)
                    np.savez_compressed(RESULTS / f"{label}_{split}.npz", probs=probs,
                                        truth=truth, ids=ids)
                    all_rows.append({"dataset": name, "scheme": scheme["key"], "seed": seed,
                                     "split": split, **metrics(probs, truth, data["classes"]),
                                     "best_source_validation_macro_f1": best,
                                     "best_epoch": best_epoch, "training_seconds": elapsed,
                                     "inference_ms_per_image": latency})
                print("TRAIN", name, scheme["key"], seed, "val", round(best, 4),
                      "epoch", best_epoch, "seconds", round(elapsed, 1), flush=True)
                del model
        (RESULTS / f"{name}_dimensions.json").write_text(json.dumps(data["dimensions"]))
    with (RESULTS / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
