#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))

import torch  # noqa: E402

from ablation_common import load_config, load_payload, model_specs  # noqa: E402
from input_rep_common import (  # noqa: E402
    EMBED_CAPTURED_LAYER, EMBED_DIR, broadcast_gain_path, dense_layer_ids, embed_vectors_path, single_layer,
)

TRAJECTORY_REL_TOL = 2e-2

BIT_EXACT_EVERY = 200

def _shard_of(context_id: str, num_shards: int) -> int:
    return int(hashlib.sha256(context_id.encode()).hexdigest(), 16) % num_shards

def _manifest_dataset_names(payload: dict) -> list[str]:
    return sorted({str(d) for d in payload["datasets"]})

def _pad_id(model):
    pad = getattr(model.generation_config, "pad_token_id", None)
    if pad is None:
        eos = model.generation_config.eos_token_id
        pad = eos[0] if isinstance(eos, (list, tuple)) else eos
    return pad

def _model_config(spec):
    from selfmorph.model import ModelConfig
    from selfmorph.settings import model_path

    cfg = load_config(spec)
    return ModelConfig(
        model_path=model_path(),
        num_layers=int(cfg["model"]["num_layers"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )

def _enumerate_samples(payload: dict):
    """(context_id, dataset, EvalSample) for EVERY payload row, enumerated from
    the frozen split manifest exactly like the oracle did.  Fails loudly if the
    manifest cannot match the payload's context set (a silent subset would
    corrupt the arm)."""
    from selfmorph.oracle import _manifest_samples

    by_ctx = {}
    for ds in _manifest_dataset_names(payload):
        for s in _manifest_samples(ds):
            by_ctx[f"{ds}::{s.qid}"] = (ds, s)
    wanted = [str(c) for c in payload["context_ids"]]
    missing = [c for c in wanted if c not in by_ctx]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(wanted)} payload contexts are not enumerated by the frozen manifest, "
            f"e.g. {missing[:5]} -- refusing to collect a silent subset"
        )
    return [(c, *by_ctx[c]) for c in wanted], by_ctx

def collect(args) -> None:
    spec = next(s for s in model_specs() if s.key == args.model)
    base_config = load_config(spec)
    payload = load_payload(spec)
    dense = dense_layer_ids(base_config)
    hidden = int(base_config["model"]["hidden_size"])
    out_dir = Path(args.out_dir) / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard{args.shard_id}of{args.num_shards}"

    success = out_dir / f"{stem}_SUCCESS.json"
    if success.is_file() and not args.force:
        print(f"[{spec.key}][{stem}] already complete -> {success}; skipping (use --force to redo)")
        return

    row_of = {str(c): i for i, c in enumerate(payload["context_ids"])}
    all_samples, by_ctx = _enumerate_samples(payload)
    my_samples = [(c, ds, s) for (c, ds, s) in all_samples
                  if _shard_of(c, args.num_shards) == args.shard_id][: (args.limit or None)]
    print(f"[{spec.key}][{stem}] {len(my_samples)}/{len(all_samples)} contexts "
          f"(manifest entries available: {len(by_ctx)})", flush=True)

    from selfmorph.benchmarks import model_wants_qwen_think_seed, render_prompt
    from selfmorph.model import get_decoder_layers, load_frozen_model

    model, tokenizer = load_frozen_model(device=args.device, model_config=_model_config(spec))
    append_think_seed = model_wants_qwen_think_seed(model)
    decoder_layers = get_decoder_layers(model)
    embed_module = model.get_input_embeddings()
    pad_id = _pad_id(model)
    print(f"[{spec.key}][{stem}] append_think_seed={append_think_seed} pad_id={pad_id} "
          f"device={args.device}", flush=True)

    gain_crosscheck: dict = {}
    if args.verify_gains:
        for arm, layer_index in (("single", single_layer(base_config) + 1), ("embed", 0)):
            path = broadcast_gain_path(arm, base_config, spec.key)
            if not path.is_file():
                gain_crosscheck[arm] = {"gain_file": str(path), "checked": False,
                                        "reason": "gain file not exported yet"}
                continue
            g = torch.load(path, map_location="cpu", weights_only=False)["gain"]
            live = decoder_layers[layer_index].input_layernorm.weight.detach().float().cpu()
            gain_crosscheck[arm] = {
                "gain_file": str(path.relative_to(Path(__file__).resolve().parents[2])),
                "live_layer_index": int(layer_index),
                "checked": True,
                "rows_equal_live_weight": bool(torch.equal(g[0], live)),
                "live_absmax": float(live.abs().max()),
            }
        print(f"[{spec.key}][{stem}] gain vs live-model cross-check: {gain_crosscheck}", flush=True)

    rows: list[dict] = []
    embeddings: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    t0 = time.time()
    for n_done, (cid, ds, sample) in enumerate(my_samples, start=1):
        prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
        enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        captured_layers: dict[int, torch.Tensor] = {}
        first_embed: list[torch.Tensor] = []
        embed_calls: list[int] = [0]
        handles = []

        def make_layer_hook(lid: int):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                captured_layers[lid] = h[:, -1, :].detach().to(torch.float32).cpu()
            return hook

        def embed_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            embed_calls[0] += 1
            if not first_embed:
                first_embed.append(h[:, -1, :].detach().to(torch.float32).cpu())

        for lid in dense:
            handles.append(decoder_layers[lid].register_forward_hook(make_layer_hook(lid)))
        handles.append(embed_module.register_forward_hook(embed_hook))
        try:
            with torch.no_grad():
                model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=1, do_sample=False, pad_token_id=pad_id,
                )
        finally:
            for h in handles:
                h.remove()

        missing_layers = [lid for lid in dense if lid not in captured_layers]
        if missing_layers or not first_embed:
            raise RuntimeError(f"{cid}: hooks missed layers={missing_layers} embed_calls={embed_calls[0]}")
        assert embed_calls[0] == 1, f"{cid}: embedding module called {embed_calls[0]} times (expected 1 prefill)"

        emb = first_embed[0].squeeze(0)
        cap = torch.stack([captured_layers[lid].squeeze(0) for lid in dense], dim=0)

        disk = payload["trajectory"][row_of[cid]]
        diff = (cap - disk).abs()
        scale = float(disk.abs().max())
        max_abs = float(diff.max())
        rel = max_abs / scale if scale > 0 else float("inf")

        g1 = float(((cap - disk).norm(dim=-1) / disk.norm(dim=-1).clamp_min(1e-9)).max())
        cos = float(torch.nn.functional.cosine_similarity(cap, disk, dim=-1).min())
        exact = bool(torch.equal(cap.half(), disk.half()))
        row = {
            "context_id": cid, "dataset": ds, "row": row_of[cid],
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "gate_max_abs": max_abs, "gate_ref_absmax": scale, "gate_rel": rel,
            "trajectory_l2_rel_max": g1, "trajectory_pass": bool(g1 < TRAJECTORY_REL_TOL),
            "gate_cos_min": cos, "gate_exact_fp16": exact,
        }

        if args.g0_every and (n_done % args.g0_every == 0 or n_done == len(my_samples)):
            from selfmorph.trajectory import collect_trajectory as _ct, stack_trajectory as _st

            prod = _st(_ct(model, enc, layer_ids=dense), dense).to(torch.float32)
            row["bit_exact"] = bool(torch.equal(cap, prod))
            row["g0_l2_rel_max"] = float(
                ((cap - prod).norm(dim=-1) / prod.norm(dim=-1).clamp_min(1e-9)).max()
            )
        rows.append(row)
        embeddings.append(emb)
        prompt_lens.append(int(enc["input_ids"].shape[1]))
        if n_done % 250 == 0 or n_done == len(my_samples):
            print(f"[{spec.key}][{stem}] {n_done}/{len(my_samples)} "
                  f"({time.time() - t0:.0f}s, worst gate_rel so far "
                  f"{max(r['gate_rel'] for r in rows):.2e})", flush=True)

    emb_t = torch.stack(embeddings, dim=0).contiguous()
    torch.save({
        "context_ids": [r["context_id"] for r in rows],
        "embeddings": emb_t,
        "prompt_lens": prompt_lens,
        "model": spec.key,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "captured_layer": EMBED_CAPTURED_LAYER,
        "embedding_module_output": True,
        "dense_layer_ids": dense,
        "hidden_size": hidden,
        "append_think_seed": bool(append_think_seed),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, out_dir / f"{stem}.pt")
    with (out_dir / f"{stem}_gate.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_exact = sum(1 for r in rows if r["gate_exact_fp16"])
    g1s = sorted(r["trajectory_l2_rel_max"] for r in rows)
    worst = max(rows, key=lambda r: r["trajectory_l2_rel_max"])
    g0_rows = [r for r in rows if "bit_exact" in r]
    g1_pass = sum(1 for r in rows if r["trajectory_pass"])
    summary = {
        "model": spec.key, "shard_id": args.shard_id, "num_shards": args.num_shards,
        "n_contexts": len(rows), "embeddings_shape": list(emb_t.shape),
        "trajectory_l2_rel_tol": TRAJECTORY_REL_TOL,
        "trajectory_l2_rel_worst": worst["trajectory_l2_rel_max"],
        "trajectory_l2_rel_median": g1s[len(g1s) // 2],
        "trajectory_l2_rel_p90": g1s[int(0.9 * len(g1s))],
        "trajectory_pass_count": g1_pass, "trajectory_pass_fraction": g1_pass / len(rows),
        "trajectory_worst_context_id": worst["context_id"],
        "gate_cos_min_over_contexts": min(r["gate_cos_min"] for r in rows),
        "bit_exact_sampled": len(g0_rows),
        "bit_exact_count": sum(1 for r in g0_rows if r["bit_exact"]),
        "bit_exact_fraction": (sum(1 for r in g0_rows if r["bit_exact"]) / len(g0_rows)) if g0_rows else None,
        "scale_normalised_max_rel": worst["gate_rel"],
        "n_exact_fp16": n_exact, "exact_fraction": n_exact / len(rows),
        "mean_prompt_tokens": sum(prompt_lens) / len(prompt_lens),
        "max_prompt_tokens": max(prompt_lens),
        "device": args.device, "elapsed_sec": time.time() - t0,
        "gain_crosscheck_vs_live_model": gain_crosscheck,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usage": "scripts/input_rep_ablation/collect_prompt_embedding.py (input-rep ablation, embed arm)",
    }
    summary["gate_pass"] = bool(
        (not g0_rows or summary["bit_exact_count"] == len(g0_rows))
        and summary["trajectory_pass_fraction"] == 1.0
    )
    with success.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{spec.key}][{stem}] DONE {len(rows)} contexts | trajectory worst={summary['trajectory_l2_rel_worst']:.2e} "
          f"median={summary['trajectory_l2_rel_median']:.2e} pass={g1_pass}/{len(rows)} | "
          f"bit-exact={summary['bit_exact_count']}/{summary['bit_exact_sampled']} | "
          f"cos_min={summary['gate_cos_min_over_contexts']:.6f} | "
          f"{'PASS' if summary['gate_pass'] else 'FAIL'} -> {success}", flush=True)
    if not summary["gate_pass"]:
        print(f"[{spec.key}][{stem}] trajectory check failed (tolerance {TRAJECTORY_REL_TOL}); data written, "
              f"but `--mode assemble` will refuse unless --allow-gate-fail is given.", flush=True)

def assemble(args) -> None:
    spec = next(s for s in model_specs() if s.key == args.model)
    payload = load_payload(spec)
    out_dir = Path(args.out_dir) / spec.key
    stems = [f"shard{i}of{args.num_shards}" for i in range(args.num_shards)]
    missing = [s for s in stems if not (out_dir / f"{s}_SUCCESS.json").is_file()]
    if missing:
        raise SystemExit(f"missing shards for {spec.key}: {missing}")

    vec_of: dict[str, torch.Tensor] = {}
    gates: list[dict] = []
    for s in stems:
        blob = torch.load(out_dir / f"{s}.pt", map_location="cpu", weights_only=False)
        for cid, vec in zip(blob["context_ids"], blob["embeddings"]):
            if cid in vec_of:
                raise SystemExit(f"duplicate context_id across shards: {cid}")
            vec_of[cid] = vec
        with (out_dir / f"{s}_gate.jsonl").open("r", encoding="utf-8") as f:
            gates.extend(json.loads(ln) for ln in f if ln.strip())

    order = [str(c) for c in payload["context_ids"]]
    missing_ids = [c for c in order if c not in vec_of]
    extra_ids = sorted(set(vec_of) - set(order))
    if missing_ids or extra_ids:
        raise SystemExit(f"assembled embeddings do not match payload contexts: "
                         f"{len(missing_ids)} missing (e.g. {missing_ids[:3]}), "
                         f"{len(extra_ids)} unexpected (e.g. {extra_ids[:3]})")
    emb = torch.stack([vec_of[c] for c in order], dim=0).contiguous()

    g1s = sorted(r["trajectory_l2_rel_max"] for r in gates)
    worst = max(gates, key=lambda r: r["trajectory_l2_rel_max"])
    n_exact = sum(1 for r in gates if r["gate_exact_fp16"])
    g0 = [r for r in gates if "bit_exact" in r]
    g1_pass = sum(1 for r in gates if r["trajectory_pass"])
    strict_ok = bool((not g0 or all(r["bit_exact"] for r in g0)) and g1_pass == len(gates))
    meta = {
        "model": spec.key, "n_contexts": len(order), "embeddings_shape": list(emb.shape),
        "dtype": str(emb.dtype), "row_order": "exactly train_dev.pt['context_ids'] (asserted)",
        "captured_layer": EMBED_CAPTURED_LAYER, "embedding_module_output": True,
        "trajectory_definition": "row-wise L2 relative error max over the 10 layers, "
                              "maximum relative L2 error across the 10 trajectory layers",
        "trajectory_l2_rel_tol": TRAJECTORY_REL_TOL,
        "trajectory_l2_rel_worst": worst["trajectory_l2_rel_max"],
        "trajectory_l2_rel_median": g1s[len(g1s) // 2],
        "trajectory_l2_rel_p90": g1s[int(0.9 * len(g1s))],
        "trajectory_pass_count": g1_pass, "trajectory_pass_fraction": g1_pass / len(gates),
        "trajectory_worst_context_id": worst["context_id"],
        "gate_cos_min_over_contexts": min(r["gate_cos_min"] for r in gates),
        "bit_exact_sampled": len(g0), "bit_exact_count": sum(1 for r in g0 if r["bit_exact"]),
        "bit_exact_fraction": (sum(1 for r in g0 if r["bit_exact"]) / len(g0)) if g0 else None,
        "n_exact_fp16": n_exact, "exact_fraction": n_exact / len(gates),
        "gate_strict_pass": strict_ok,
        "gate_override_allow_fail": bool(args.allow_gate_fail),
        "n_shards": args.num_shards,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(meta, indent=2, default=str))
    if not strict_ok and not args.allow_gate_fail:
        raise SystemExit(
            f"ALIGNMENT GATE FAILED on the assembled payload for {spec.key} "
            f"(trajectory worst {meta['trajectory_l2_rel_worst']:.3e} > {TRAJECTORY_REL_TOL}, pass "
            f"{g1_pass}/{len(gates)}; bit-exact {meta['bit_exact_count']}/{meta['bit_exact_sampled']}). "
            f"Re-run with --allow-gate-fail to override EXPLICITLY (the override is recorded in the metadata)."
        )
    torch.save({"context_ids": order, "embeddings": emb, "meta": meta}, embed_vectors_path(spec))
    with (EMBED_DIR / spec.key / "prompt_embedding_aligned.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"saved {embed_vectors_path(spec)}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["qwen35_27b", "qwen35_4b", "gemma4_12b"])
    ap.add_argument("--mode", default="collect", choices=["collect", "assemble"])
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=str(EMBED_DIR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N contexts of this shard")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bit-exact-every", type=int, default=BIT_EXACT_EVERY,
                    help="compare with the probing trajectory every N contexts; 0 disables the check")
    ap.add_argument("--allow-gate-fail", action="store_true",
                    help="assemble: proceed even though the strict alignment gate failed (recorded in metadata)")
    ap.add_argument("--no-verify-gains", dest="verify_gains", action="store_false")
    args = ap.parse_args()
    if args.mode == "collect":
        if not 0 <= args.shard_id < args.num_shards:
            raise SystemExit("--shard-id must be in [0, --num-shards)")
        collect(args)
    else:
        assemble(args)

if __name__ == "__main__":
    main()
