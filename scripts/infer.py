#!/usr/bin/env python3
"""Run Probe–Select–Morph on text prompts or one benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.benchmarks import load_benchmark, render_prompt  # noqa: E402
from selfmorph.benchmarks.prompts import model_wants_qwen_think_seed  # noqa: E402
from selfmorph.executor import execute_anchor  # noqa: E402
from selfmorph.model import ModelConfig, load_frozen_model  # noqa: E402
from selfmorph.selector import build_selector  # noqa: E402
from selfmorph.settings import PACKAGE_ROOT, model_path  # noqa: E402
from selfmorph.trajectory import load_trajectory_spec, probe_with_prefix, stack_trajectory  # noqa: E402


def _read_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            lines.append(text)
    return lines


def _inputs(args) -> list[tuple[str | None, object | None, str]]:
    """Return (dataset, sample_or_none, prompt_text) rows."""
    rows: list[tuple[str | None, object | None, str]] = []
    if args.dataset:
        samples = load_benchmark(args.dataset)
        if args.limit is not None:
            samples = samples[: args.limit]
        for sample in samples:
            rows.append((args.dataset, sample, ""))
        return rows
    texts = list(args.text)
    if args.input_file:
        texts.extend(_read_lines(Path(args.input_file)))
    if not texts and not sys.stdin.isatty():
        texts.extend(line.strip() for line in sys.stdin if line.strip())
    if args.limit is not None:
        texts = texts[: args.limit]
    return [(None, None, text) for text in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime.json")
    parser.add_argument("--model-path", default=None, help="Local path or Hugging Face repo id")
    parser.add_argument("--selector-checkpoint", default=None)
    parser.add_argument("--dataset", default=None, help="Benchmark name, for example gsm8k or bbh:formal_fallacies")
    parser.add_argument("--text", action="append", default=[], help="One prompt; repeat for several")
    parser.add_argument("--input-file", default=None, help="UTF-8 file with one prompt per line")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="JSONL path; default is stdout")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PACKAGE_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))

    rows = _inputs(args)
    if not rows:
        parser.error("pass --dataset, --text, --input-file, or prompts on stdin")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = args.model_path or model_path()
    model_cfg = ModelConfig(
        model_path=backbone,
        num_layers=config["model"]["num_layers"],
        hidden_size=config["model"]["hidden_size"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    model, tokenizer = load_frozen_model(device=device, model_config=model_cfg)
    append_think_seed = model_wants_qwen_think_seed(model)

    anchors = build_action_space(config)
    layer_ids = load_trajectory_spec(config).layer_ids()
    probe_depth = max(layer_ids) + 1
    repeat_exits = {int(a.layer_span[1]) for a in anchors if a.type == "REPEAT"}
    prefix_depths = tuple(sorted(repeat_exits | {probe_depth}))

    ckpt = args.selector_checkpoint or config["selector"]["selector_checkpoint"]
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PACKAGE_ROOT / ckpt_path
    selector = build_selector(
        config["selector"], anchors, config["model"]["hidden_size"], len(layer_ids)
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    selector.load_state_dict(state["model_state"])
    selector.eval()

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for index, (dataset, sample, text) in enumerate(rows):
            max_new = args.max_new if args.max_new is not None else 1024
            if sample is not None:
                prompt = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
                source = f"{dataset}::{sample.qid}"
            else:
                prompt = render_prompt(
                    tokenizer, [{"role": "user", "content": text}], append_think_seed=append_think_seed,
                )
                source = str(index)
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            probe = probe_with_prefix(model, enc, layer_ids=layer_ids, prefix_depths=prefix_depths)
            trajectory = stack_trajectory(probe["hiddens"], layer_ids).float().unsqueeze(0).to(device)
            with torch.no_grad():
                hidden = selector(trajectory)
                anchor_id = int(selector.head.joint_logp_anchors(hidden, anchors).argmax())
            anchor = anchors.by_id[anchor_id]
            generated = execute_anchor(
                model, tokenizer, enc, anchor,
                num_layers=config["model"]["num_layers"],
                action_cfg=config["action_space"],
                max_new_tokens=max_new,
                prefix=probe["prefix"],
            )
            record = {
                "id": source,
                "anchor_id": anchor_id,
                "type": anchor.type,
                "layer_span": list(anchor.layer_span) if anchor.layer_span else None,
                "r": anchor.r,
                "K": anchor.K,
                "text": generated["text"],
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
