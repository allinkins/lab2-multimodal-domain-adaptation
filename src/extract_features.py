"""Extract frozen SigLIP 2 embeddings and direct image statistics.

The text is the existing Apple Vision description for Office-Home, or the
class-free geometry description for Digits. No ground-truth label is passed to
the encoder. Numeric features are measured once on the original displayed
image; there is no artificial time axis.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.datasets import load_digits
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LAB1 = WORKSPACE / "lab1_officehome"
MODEL_DIR = Path(os.environ.get("SIGLIP2_MODEL_DIR", ROOT / ".model_cache/siglip2"))
MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
NUMERIC_NAMES = [
    "red_mean", "green_mean", "blue_mean", "red_sd", "green_sd", "blue_sd",
    "saturation_mean", "saturation_sd", "value_mean", "value_sd",
    "gray_mean", "gray_sd", "gray_q10", "gray_q90", "dark_fraction",
    "bright_fraction", "mean_gradient", "edge_fraction", "log_aspect_ratio",
]


def digit_image(raw: np.ndarray, target: bool) -> Image.Image:
    gray = np.uint8(np.clip(raw / 16.0 * 255, 0, 255))
    image = Image.fromarray(gray).resize((64, 64), Image.Resampling.BICUBIC)
    if target:
        image = ImageEnhance.Contrast(image).enhance(1.25)
        arr = 255 - np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(np.stack([arr, np.uint8(arr * .83),
                                          np.uint8(arr * .66)], axis=2)).filter(
                                              ImageFilter.GaussianBlur(.5))
    return image.convert("RGB")


def image_for(dataset: str, row: dict, digits_raw: np.ndarray | None) -> Image.Image:
    if dataset == "officehome":
        # Rows and source manifest have the same stable ids.
        with Image.open(LAB1 / "data/images" / f"{row['id']:05d}.jpg") as source:
            return source.convert("RGB")
    return digit_image(digits_raw[row["id"]], row["split"] in {"target_adapt", "test"})


def numeric_features(image: Image.Image) -> np.ndarray:
    width, height = image.size
    rgb_image = image.resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(rgb_image.convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gradient = (gx[:-1, :] + gy[:, :-1]) / 2.0
    result = [*rgb.mean((0, 1)), *rgb.std((0, 1)), hsv[:, :, 1].mean(),
              hsv[:, :, 1].std(), hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
              gray.mean(), gray.std(), *np.quantile(gray, [.1, .9]),
              (gray < .15).mean(), (gray > .85).mean(), gradient.mean(),
              (gradient > .08).mean(), np.log(width / height)]
    assert len(result) == len(NUMERIC_NAMES)
    return np.asarray(result, dtype=np.float32)


def normalize(x: torch.Tensor) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        x = x.pooler_output
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


def encode_texts(model, processor, texts: list[str], batch: int) -> np.ndarray:
    unique = list(dict.fromkeys(texts))
    lookup: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), batch):
        block = unique[start:start + batch]
        inputs = processor(text=block, padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt")
        with torch.inference_mode():
            vectors = normalize(model.get_text_features(**inputs))
        lookup.update(zip(block, vectors))
        print(f"TEXT {start + len(block)}/{len(unique)}", flush=True)
    return np.stack([lookup[t] for t in texts])


def class_prompts(dataset: str, rows: list[dict]) -> list[str]:
    if dataset == "officehome":
        manifest = json.loads((LAB1 / "data/manifest.json").read_text())
        mapping = {int(r["label"]): r["class_name"].replace("_", " ") for r in manifest}
        return [f"a photo of a {mapping[i].lower()}" for i in range(65)]
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return [f"a handwritten digit {name}" for name in names]


def run(dataset: str, batch: int, limit: int | None):
    rows = json.loads((ROOT / f"data/{dataset}_rows.json").read_text())
    if limit is not None:
        rows = rows[:limit]
    torch.set_num_threads(8)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True,
                                       use_safetensors=True).eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    digits_raw = load_digits().images if dataset == "digits" else None
    visual: list[np.ndarray] = []
    numeric: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch):
        block = rows[start:start + batch]
        images = [image_for(dataset, r, digits_raw) for r in block]
        numeric.extend(numeric_features(img) for img in images)
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            visual.extend(normalize(model.get_image_features(**inputs)))
        for img in images:
            img.close()
        if start == 0 or (start + len(block)) % 100 < batch or start + len(block) == len(rows):
            elapsed = time.perf_counter() - started
            print(f"IMAGE {start + len(block)}/{len(rows)} seconds={elapsed:.1f}", flush=True)
    texts = [r["text"] for r in rows]
    text = encode_texts(model, processor, texts, max(batch, 16))
    prompts = class_prompts(dataset, rows)
    prototypes = encode_texts(model, processor, prompts, max(batch, 16))
    visual_array = np.stack(visual).astype(np.float32)
    numeric_array = np.stack(numeric).astype(np.float32)
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    dest = ROOT / "data"
    dest.mkdir(exist_ok=True)
    suffix = "_sample" if limit is not None else ""
    np.savez_compressed(dest / f"{dataset}_features{suffix}.npz", visual=visual_array,
                        text=text.astype(np.float32), numeric=numeric_array,
                        prototypes=prototypes.astype(np.float32), ids=ids)
    (dest / f"{dataset}_rows{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    meta = {"model_id": MODEL_ID, "revision": REVISION, "dataset": dataset,
            "n": len(rows), "visual_dim": visual_array.shape[1],
            "text_dim": text.shape[1], "numeric_names": NUMERIC_NAMES,
            "numeric_from": "single original displayed image; no temporal sequence",
            "text_from": "Apple Vision labels" if dataset == "officehome" else
                         "deterministic class-free geometry description",
            "seconds_total": time.perf_counter() - started,
            "seconds_per_image_and_text_average": (time.perf_counter() - started) / len(rows),
            "device": "cpu", "logit_scale": float(model.logit_scale.exp().detach())}
    (dest / f"{dataset}_extraction{suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["officehome", "digits"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run(args.dataset, args.batch, args.limit)
