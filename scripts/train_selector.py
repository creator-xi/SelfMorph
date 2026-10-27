#!/usr/bin/env python3

"""Train a morphology selector from a cached oracle tensor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfmorph.training import train_selector

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to runtime.json")
    parser.add_argument("--train-dev-pt", required=True, help="Path to train_dev.pt from build_training_data.py")
    parser.add_argument("--output-dir", required=True, help="Output dir for history.jsonl / best_checkpoint.pt")
    parser.add_argument("--device", default="cpu", help="Training device (cached-feature training is cheap; CPU is fine)")
    parser.add_argument("--seed", type=int, default=42, help="Init seed written into selector.seed (default 42)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["selector"]["seed"] = args.seed

    result = train_selector(train_dev_pt=Path(args.train_dev_pt), output_dir=Path(args.output_dir), config=config, device=args.device)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
