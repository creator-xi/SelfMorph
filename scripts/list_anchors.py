#!/usr/bin/env python3
"""List the anchors enumerated by config/runtime.json's action_space
section. Usage:
    python scripts/list_anchors.py --config config/runtime.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfmorph.action_space import build_action_space

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to runtime.json")
    parser.add_argument("--json", action="store_true", help="Print as JSON instead of a table")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    anchors = build_action_space(config)

    if args.json:
        print(json.dumps([
            {"id": a.id, "type": a.type, "layer_span": a.layer_span, "r": a.r, "K": a.K, "K_eff": a.K_eff, "analytic_cost": a.analytic_cost}
            for a in anchors
        ], indent=2))
        return

    print(f"{len(anchors)} anchors (num_layers={anchors.num_layers})")
    print(f"{'id':>3} {'type':<10} {'layer_span':<12} {'r/K':>4} {'analytic_cost':>14}")
    for a in anchors:
        theta = a.r if a.type == "REPEAT" else (a.K if a.type == "BRANCH" else "-")
        span = str(a.layer_span) if a.layer_span else "-"
        print(f"{a.id:>3} {a.type:<10} {span:<12} {theta!s:>4} {a.analytic_cost:>14.4f}")

if __name__ == "__main__":
    main()
