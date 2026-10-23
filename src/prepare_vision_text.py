"""Join saved Apple Vision labels to Office-Home objects by stable image id."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB1 = ROOT.parent / "lab1_officehome"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=LAB1 / "data/manifest.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/officehome_vision_labels.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/officehome_rows.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    observations = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    by_id = {int(row["id"]): row for row in observations}
    if len(by_id) != len(observations) or set(by_id) != {int(row["id"]) for row in manifest}:
        raise ValueError("Apple Vision ids must be unique and match every manifest image")

    rows = []
    for item in manifest:
        source = by_id[int(item["id"])]
        if source.get("error") or not source.get("labels"):
            raise ValueError(f"Apple Vision did not classify image {item['id']}")
        text = " ".join(str(tag["label"]).replace("_", " ") for tag in source["labels"][:6])
        rows.append({key: item[key] for key in ("id", "split", "domain", "label", "group")}
                    | {"text": text})
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
