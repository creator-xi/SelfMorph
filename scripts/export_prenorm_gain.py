#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_KEY_TEMPLATE = "model.language_model.layers.{L}.input_layernorm.weight"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True,
                        help="Frozen base model dir containing the safetensors file.")
    parser.add_argument("--safetensors", default="model.safetensors",
                        help="safetensors file name inside --model-dir (default: model.safetensors).")
    parser.add_argument("--key-template", default=DEFAULT_KEY_TEMPLATE,
                        help="state-dict key template; must contain '{L}' (the CONSUMER layer id).")
    parser.add_argument("--captured-layers", required=True,
                        help="Comma-separated captured layer ids (the trajectory layer set), in canonical order.")
    parser.add_argument("--consumer-offset", type=int, default=1,
                        help="Consumer block = captured layer + this offset (default 1: hidden state at "
                             "layer l's output is consumed by block l+1).")
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument("--source", default=None,
                        help="Free-text provenance string recorded in the file (default: auto-generated).")
    parser.add_argument("--note", default="", help="Free-text note recorded in the file.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing output file.")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    captured = [int(x) for x in args.captured_layers.replace(",", " ").split()]
    if not captured:
        raise SystemExit("--captured-layers must not be empty")
    next_layers = [l + args.consumer_offset for l in captured]

    st_path = Path(args.model_dir) / args.safetensors
    if not st_path.is_file():
        raise SystemExit(f"safetensors not found: {st_path}")

    rows = []
    keys = []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        available = set(f.keys())
        for L in next_layers:
            key = args.key_template.format(L=L)
            if key not in available:
                raise SystemExit(f"key not found in {st_path.name}: {key}")
            t = f.get_tensor(key)
            if t.dim() != 1:
                raise SystemExit(f"unexpected shape for {key}: {tuple(t.shape)} (expected 1-D gain)")
            rows.append(t.to(torch.float32))
            keys.append(key)

    shapes = {tuple(r.shape) for r in rows}
    if len(shapes) != 1:
        raise SystemExit(f"inconsistent per-layer gain shapes: {sorted(shapes)}")
    gain = torch.stack(rows, dim=0).contiguous()
    if gain.shape[0] != len(captured):
        raise SystemExit("internal error: stacked gain row count != len(captured_layers)")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file (pass --force to allow): {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = args.source or (
        f"{st_path} ({args.key_template})"
    )
    payload = {
        "gain": gain,
        "captured_layers": captured,
        "next_layers": next_layers,
        "source": source,
        "rms_norm_eps": args.rms_norm_eps,
        "note": args.note,
    }
    torch.save(payload, out_path)

    print(f"wrote {out_path}")
    print(f"  gain            : shape={tuple(gain.shape)} dtype={gain.dtype}")
    print(f"  captured_layers : {captured}")
    print(f"  next_layers     : {next_layers}")
    print(f"  rms_norm_eps    : {args.rms_norm_eps}")
    print(f"  source          : {source}")
    print("  per-row fingerprint (first 3 values + mean/std):")
    for i, (L, key) in enumerate(zip(next_layers, keys)):
        r = gain[i]
        print(f"    [{i:2d}] captured={captured[i]:2d} <- consumer L={L:2d}  "
              f"head={[round(float(v), 6) for v in r[:3]]} "
              f"mean={float(r.mean()):.6f} std={float(r.std()):.6f}")
    print("  reload check:", end=" ")
    back = torch.load(out_path, map_location="cpu", weights_only=False)
    ok = (
        torch.equal(back["gain"], gain)
        and list(back["captured_layers"]) == captured
        and list(back["next_layers"]) == next_layers
    )
    print("OK" if ok else "MISMATCH")
    if not ok:
        sys.exit(1)
    print(json.dumps({"out": str(out_path), "shape": list(gain.shape)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
