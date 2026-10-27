#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfmorph.training_data import build_training_data, save_training_data

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to runtime.json")
    parser.add_argument("--oracle-glob", required=True, help="Glob of oracle jsonl shards")
    parser.add_argument("--out-dir", required=True, help="Output directory for train_dev.pt + contexts.jsonl")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    records, anchors = build_training_data(oracle_glob=args.oracle_glob, config=config)

    out_dir = Path(args.out_dir)
    out_pt = out_dir / "train_dev.pt"
    out_contexts = out_dir / "contexts.jsonl"
    save_training_data(
        records, anchors, out_pt_path=out_pt, out_contexts_path=out_contexts,
        include_math_in_oracle=config["domains"]["include_math_in_oracle"],
    )
    print(f"Wrote {len(records)} contexts -> {out_pt}")
    from collections import Counter

    print("Split counts:", Counter(r.split for r in records))
    print("Domain counts:", Counter(r.domain for r in records))

if __name__ == "__main__":
    main()
