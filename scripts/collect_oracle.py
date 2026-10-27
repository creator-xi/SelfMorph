#!/usr/bin/env python3
"""Collect the 25-action oracle and hidden trajectories for one shard.

Launch the full run with scripts/run_collect_oracle.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfmorph.model import load_frozen_model
from selfmorph.oracle import assert_no_duplicate_process, collect_oracle_shard

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to runtime.json")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names (see benchmarks/datasets.py)")
    parser.add_argument("--output-dir", required=True, help="Base output dir; writes to <output-dir>/shard{i}of{n}/oracle.jsonl")
    parser.add_argument("--shard-id", type=int, required=True, help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--device", default="cuda", help="Device for the frozen base LM")
    parser.add_argument("--fsync-every", type=int, default=1000, help="fsync every N records")
    parser.add_argument("--collect-mode", default="full", choices=["full", "br_only", "rep_only"],
                        help="full collects all 25 actions and the trajectory; "
                             "br_only collects BRANCH; rep_only collects REPEAT")
    parser.add_argument("--context-ids-file", default=None,
                        help="Optional whitelist, one dataset::qid per line")
    args = parser.parse_args()

    assert_no_duplicate_process("scripts/collect_oracle.py", args.shard_id)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, tokenizer = load_frozen_model(device=args.device)
    collect_oracle_shard(
        dataset_names=args.datasets.split(","),
        output_dir=Path(args.output_dir),
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        config=config,
        model=model,
        tokenizer=tokenizer,
        fsync_every=args.fsync_every,
        collect_mode=args.collect_mode,
        context_ids_file=Path(args.context_ids_file) if args.context_ids_file else None,
    )

if __name__ == "__main__":
    main()
