#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from ablation_common import (  # noqa: E402
    load_config, load_payload, model_specs, slice_payload_trajectory,
)

KEY_TEMPLATE = "model.language_model.layers.{L}.input_layernorm.weight"
GAIN_DIR = REPO_ROOT / "resources" / "base_model_prenorm"

def _resolve_shards(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    from safetensors import safe_open

    mapping: dict[str, str] = {}
    for shard in sorted(p.name for p in model_dir.glob("*.safetensors")):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                mapping[key] = shard
    return mapping

def build_gain(captured_layers: list[int], model_dir: Path) -> torch.Tensor:
    from safetensors import safe_open

    weight_map = _resolve_shards(model_dir)
    wanted: list[tuple[int, str, str]] = []
    for l in captured_layers:
        key = KEY_TEMPLATE.format(L=int(l) + 1)
        shard = weight_map.get(key)
        if shard is None:
            raise SystemExit(f"key not found in {model_dir.name} index/headers: {key}")
        wanted.append((int(l) + 1, key, shard))

    rows = []
    for shard in sorted({s for _, _, s in wanted}):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for _, key, s in wanted:
                if s != shard:
                    continue
                t = f.get_tensor(key)
                if t.dim() != 1:
                    raise SystemExit(f"unexpected shape for {key}: {tuple(t.shape)}")
                rows.append(t.to(torch.float32).clone())
    return torch.stack(rows, dim=0).contiguous()

def gain_path_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))

ARMS: tuple[str, ...] = ("embed", "single", "layers3", "layers5", "traj10")

ARM_INPUT_LABEL: dict[str, str] = {
    "embed": "Prompt embedding",
    "single": "Single hidden state",
    "layers3": "3 layers",
    "layers5": "5 layers",
    "traj10": "Hidden trajectory",
}

SINGLE_SLOT: int = 4

ARM_SLOT_SETS: dict[str, tuple[int, ...]] = {"layers5": (0, 2, 4, 6, 8), "layers3": (0, 4, 8)}

EMBED_CAPTURED_LAYER: int = -1

BROADCAST_ARMS: tuple[str, ...] = ("single", "embed")

CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "input_rep_ablation"
EMBED_DIR = REPO_ROOT / "artifacts" / "results" / "input_rep_ablation" / "embeddings"
TRAJ10_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "trajectory_count_ablation"
DEFAULT_PRENORM_GAIN_NAME = "qwen35_27b_input_layernorm_gain.pt"

def dense_layer_ids(base_config: dict) -> list[int]:
    return [int(x) for x in base_config["trajectory"]["dense_layer_ids"]]

def single_layer(base_config: dict) -> int:
    """The physical layer broadcast by the `single` arm (slot `SINGLE_SLOT`)."""
    dense = dense_layer_ids(base_config)
    if not 0 <= SINGLE_SLOT < len(dense):
        raise ValueError(f"trajectory has {len(dense)} slots, SINGLE_SLOT={SINGLE_SLOT} is invalid")
    return dense[SINGLE_SLOT]

def replicate_to_slots(selected: tuple[int, ...], n_slots: int = 10) -> list[int]:
    if not selected:
        raise ValueError("selected slot set must not be empty")
    out: list[int] = []
    for i in range(n_slots):
        j = min(range(len(selected)), key=lambda j: (abs(i - selected[j]), j))
        out.append(int(selected[j]))
    return out

def arm_slot_map(arm: str) -> list[int] | None:
    """The 10 trajectory-slot indices the arm reads (None for `embed`, whose
    input is not a trajectory slot, and for `traj10`, which reads them as-is)."""
    if arm == "single":
        return [SINGLE_SLOT] * 10
    if arm in ARM_SLOT_SETS:
        return replicate_to_slots(ARM_SLOT_SETS[arm])
    return None

def captured_layers_for(arm: str, base_config: dict) -> list[int]:
    """`captured_layers` metadata for the arm's gain file (also stored in the
    trial's `selector.expected_trajectory_layer_ids`, which makes
    `resolve_prenorm_gain` fail loudly on a layer-set mismatch)."""
    if arm == "single":
        return [single_layer(base_config)] * 10
    if arm == "embed":
        return [EMBED_CAPTURED_LAYER] * 10
    if arm in ARM_SLOT_SETS:
        dense = dense_layer_ids(base_config)
        return [dense[i] for i in arm_slot_map(arm)]
    raise ValueError(f"arm {arm!r} does not need a broadcast gain (traj10 keeps the production gain)")

def authoritative_gain_path(base_config: dict) -> Path:
    """The production (authoritative) 10-row gain file of this base, resolved
    exactly like `build_selector` does."""
    p = base_config["selector"].get("prenorm_gain_path")
    return (REPO_ROOT / p) if p else (REPO_ROOT / "resources" / "base_model_prenorm" / DEFAULT_PRENORM_GAIN_NAME)

def broadcast_gain_name(arm: str, base_config: dict, spec_key: str) -> str:
    if arm == "single":
        return f"{spec_key}_layer{single_layer(base_config)}_input_layernorm_gain.pt"
    if arm == "embed":
        return "%s_prompt_embedding_layernorm_gain.pt" % spec_key
    if arm in ARM_SLOT_SETS:
        return f"{spec_key}_layers{len(ARM_SLOT_SETS[arm])}_input_layernorm_gain.pt"
    raise ValueError(f"arm {arm!r} has no broadcast gain")

def broadcast_gain_path(arm: str, base_config: dict, spec_key: str) -> Path:
    return GAIN_DIR / broadcast_gain_name(arm, base_config, spec_key)

def export_broadcast_gain(spec, base_config: dict, arm: str, *, force: bool = False) -> Path:
    """Export `resources/base_model_prenorm/<broadcast_gain_name>` (NEW file;
    never overwrites a production gain) and verify it by reload."""
    captured = captured_layers_for(arm, base_config)
    hidden = int(base_config["model"]["hidden_size"])
    out_path = broadcast_gain_path(arm, base_config, spec.key)

    if out_path.is_file() and not force:
        back = torch.load(out_path, map_location="cpu", weights_only=False)
        if (list(back.get("captured_layers", [])) == captured
                and tuple(back["gain"].shape) == (len(captured), hidden)):
            return out_path

    from selfmorph.settings import model_path
    model_dir = Path(model_path())
    if not model_dir.is_dir():
        raise SystemExit(f"base model dir not found: {model_dir}")
    gain = build_gain(captured, model_dir)
    if tuple(gain.shape) != (len(captured), hidden):
        raise SystemExit(f"gain shape {tuple(gain.shape)} != ({len(captured)}, {hidden})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gain": gain,
        "captured_layers": captured,
        "next_layers": [l + 1 for l in captured],
        "source": f"{model_dir} ({KEY_TEMPLATE})",
        "rms_norm_eps": 1e-6,
        "note": (f"INPUT-REPRESENTATION ablation arm {arm!r}: the selector input is one vector "
                 f"broadcast to all 10 trajectory slots, so its pre-norm gain is the SAME row "
                 f"repeated 10 times (captured layer {'-1 = prompt embedding' if arm == 'embed' else captured[0]}"
                 f" consumed by block {'0' if arm == 'embed' else captured[0] + 1}). Exported by "
                 f"scripts/input_rep_ablation/input_rep_common.py (shard-aware; same payload schema as "
                 f"scripts/export_prenorm_gain.py)."),
    }, out_path)

    back = torch.load(out_path, map_location="cpu", weights_only=False)
    if not (torch.equal(back["gain"], gain) and list(back["captured_layers"]) == captured):
        raise SystemExit(f"gain reload check FAILED for {out_path}")
    return out_path

def verify_broadcast_gain(path: Path, base_config: dict, arm: str, *, auth_payload: dict | None = None) -> dict:
    """Acceptance evidence for a broadcast gain file.

    Checks: shape `(10, H)`; all 10 rows BIT-IDENTICAL; metadata matches the
    design (`captured_layers == [L]*10`, `next_layers == [L+1]*10`); and -- for
    the `single` arm, whose layer IS present in the authoritative 10-row
    production file -- the row is bit-identical to that file's matching row
    (cross-check that the right layer was picked).  For `embed` (layer -1,
    absent from every existing gain file) the row is instead cross-checked
    against the live loaded model by `collect_prompt_embedding.py`; here we only
    require that it DIFFERS from every authoritative row (it must not be a
    stale/duplicated production row).
    """
    hidden = int(base_config["model"]["hidden_size"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gain = payload["gain"]
    out: dict = {
        "path": str(path.relative_to(REPO_ROOT)), "arm": arm,
        "shape": list(gain.shape), "captured_layers": list(payload["captured_layers"]),
        "next_layers": list(payload["next_layers"]), "dtype": str(gain.dtype),
        "rows_bit_identical": bool(torch.equal(gain, gain[0:1].expand_as(gain))),
        "row0_absmax": float(gain[0].abs().max()),
    }
    expected_captured = captured_layers_for(arm, base_config)
    out["metadata_matches_design"] = (
        list(payload["captured_layers"]) == expected_captured
        and list(payload["next_layers"]) == [l + 1 for l in expected_captured]
    )
    out["shape_ok"] = tuple(gain.shape) == (10, hidden)

    auth_path = authoritative_gain_path(base_config)
    auth = auth_payload if auth_payload is not None else torch.load(auth_path, map_location="cpu", weights_only=False)
    auth_gain = auth["gain"]
    auth_layers = [int(x) for x in auth["captured_layers"]]
    captured = [int(x) for x in payload["captured_layers"]]
    out["authoritative_gain_path"] = str(auth_path.relative_to(REPO_ROOT))
    out["authoritative_captured_layers"] = auth_layers
    out["design_slot_map"] = arm_slot_map(arm)

    provenance = []
    for l in range(len(captured)):
        cl = captured[l]
        idx = auth_layers.index(cl) if cl in auth_layers else None
        provenance.append({
            "slot": l, "captured_layer": cl, "authoritative_row_index": idx,
            "matches_authoritative": (idx is not None) and bool(torch.equal(gain[l], auth_gain[idx])),
        })
    out["row_provenance"] = provenance
    out["all_rows_match_authoritative"] = all(p["matches_authoritative"] for p in provenance)
    out["n_distinct_rows"] = len({gain[l].numpy().tobytes() for l in range(len(captured))})
    out["n_distinct_captured_layers"] = len(set(captured))

    if arm in BROADCAST_ARMS:
        if arm == "single":
            L = single_layer(base_config)
            if L in auth_layers:
                i = auth_layers.index(L)
                out["row_equals_authoritative_row"] = bool(torch.equal(gain[0], auth_gain[i]))
                out["authoritative_row_index"] = i
            else:
                out["row_equals_authoritative_row"] = None
                out["authoritative_row_index"] = None
        else:
            out["differs_from_every_authoritative_row"] = all(
                not torch.equal(gain[0], auth_gain[i]) for i in range(len(auth_layers))
            )
        out["ok"] = bool(
            out["shape_ok"] and out["rows_bit_identical"] and out["metadata_matches_design"]
            and (out.get("row_equals_authoritative_row") is not False)
            and (out.get("differs_from_every_authoritative_row") is not False)
        )
    else:
        out["row_runs_match_slot_map"] = all(
            provenance[l]["captured_layer"] == captured_layers_for(arm, base_config)[l] for l in range(10)
        )
        out["row_pattern_is_designed_replication"] = bool(
            out["n_distinct_rows"] == out["n_distinct_captured_layers"]
            and [p["captured_layer"] for p in provenance] == captured_layers_for(arm, base_config)
        )
        out["ok"] = bool(
            out["shape_ok"] and out["metadata_matches_design"] and out["all_rows_match_authoritative"]
            and out["row_pattern_is_designed_replication"]
        )
    return out

def embed_vectors_path(spec) -> Path:
    return EMBED_DIR / spec.key / "prompt_embedding_aligned.pt"

def load_embed_vectors(spec) -> dict:
    path = embed_vectors_path(spec)
    if not path.is_file():
        raise SystemExit(
            f"aligned prompt embeddings not found: {path}\n"
            f"run scripts/input_rep_ablation/collect_prompt_embedding.py --model {spec.key} --mode assemble first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)

def build_embed_payload(raw_payload: dict, spec) -> dict:
    """`[n, 10, H]` broadcast embed payload, ROW-ALIGNED to `train_dev.pt`."""
    blob = load_embed_vectors(spec)
    vec = blob["embeddings"]
    ids = list(blob["context_ids"])
    raw_ids = [str(c) for c in raw_payload["context_ids"]]
    if ids != raw_ids:
        n_bad = sum(1 for a, b in zip(ids, raw_ids) if a != b)
        raise ValueError(
            f"embed payload row order does not match train_dev.pt context_ids "
            f"({n_bad} mismatching rows of {len(raw_ids)}; first mismatch "
            f"{next(((a, b) for a, b in zip(ids, raw_ids) if a != b), None)})"
        )
    if tuple(vec.shape) != (len(raw_ids), int(raw_payload["trajectory"].shape[2])):
        raise ValueError(f"embedding shape {tuple(vec.shape)} != ({len(raw_ids)}, H)")
    t = vec.unsqueeze(1).expand(-1, 10, -1).contiguous()
    out = dict(raw_payload)
    out["trajectory"] = t
    out["input_rep_arm"] = "embed"
    return out

def build_arm_payload(raw_payload: dict, spec, arm: str) -> dict:
    """The ONLY thing an arm changes: the selector's input tensor."""
    if arm == "traj10":
        return dict(raw_payload)
    if arm == "single" or arm in ARM_SLOT_SETS:
        payload = slice_payload_trajectory(raw_payload, arm_slot_map(arm))
        payload["input_rep_arm"] = arm
        return payload
    if arm == "embed":
        return build_embed_payload(raw_payload, spec)
    raise ValueError(f"unknown arm {arm!r}")

def build_identity_payload_spec() -> dict:
    """`traj_subset` spec that is the IDENTITY on the layer axis -- used when a
    worker is handed the embed payload FILE directly; `ablation_common`'s own
    payload builder then keeps that tensor unchanged.
    """
    return {"kind": "traj_subset", "layer_idx": list(range(10))}

def build_trials(spec, base_config: dict, arms: list[str]) -> tuple[list[dict], dict]:
    """Trials for one base.  `traj10` keeps `selector` EXACTLY as production
    (self-check arm); the two broadcast arms only override
    `prenorm_gain_path` + `expected_trajectory_layer_ids`."""
    dense = dense_layer_ids(base_config)
    gains: dict[str, str] = {}
    for arm in arms:
        if arm in BROADCAST_ARMS or arm in ARM_SLOT_SETS:
            gains[arm] = gain_path_relative(export_broadcast_gain(spec, base_config, arm))

    trials: list[dict] = []
    for arm in arms:
        slots = arm_slot_map(arm)
        selector = copy.deepcopy(base_config["selector"])
        if arm in BROADCAST_ARMS or arm in ARM_SLOT_SETS:
            selector["prenorm_gain_path"] = gains[arm]
            selector["expected_trajectory_layer_ids"] = captured_layers_for(arm, base_config)
        trials.append({
            "trial_id": arm,
            "model": spec.key,
            "arm": arm,
            "input_label": ARM_INPUT_LABEL[arm],
            "num_trajectory_slots": 10,
            "trajectory_slot_indices": (slots if slots is not None
                                        else (list(range(10)) if arm == "traj10" else "broadcast(embed)")),
            "trajectory_layer_ids": ([dense[i] for i in slots] if slots is not None
                                     else (dense if arm == "traj10" else "prompt_embedding(layer -1)")),
            "selected_slots_0based": list(ARM_SLOT_SETS.get(arm, ())) or None,
            "selector_config": selector,
            "payload_spec": {"kind": "identity", "note": "payload is built by this module's worker, not by ablation_common"},
        })
    meta = {
        "dense_layer_ids": dense,
        "arm_input_labels": {a: ARM_INPUT_LABEL[a] for a in arms},
        "single_slot_index": SINGLE_SLOT,
        "single_layer": single_layer(base_config),
        "embed_captured_layer": EMBED_CAPTURED_LAYER,
        "arm_slot_maps": {a: arm_slot_map(a) for a in arms},
        "arm_selected_slots_0based": {a: list(ARM_SLOT_SETS[a]) for a in arms if a in ARM_SLOT_SETS},
        "arm_layers": {a: ([dense[i] for i in arm_slot_map(a)] if arm_slot_map(a) is not None else None) for a in arms},
        "gain_paths": gains,
    }
    return trials, meta

def train_worker_arm(worker_trials: list[dict], base_config: dict, train_dev_pt: str, output_dir: str,
                     device: str, result_queue) -> None:
    import torch as _torch

    from selfmorph.training import train_selector
    from selfmorph.training_data import merge_dev_splits

    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True

    raw_payload = _torch.load(train_dev_pt, map_location="cpu", weights_only=False)
    spec = next(s for s in model_specs() if s.key == worker_trials[0]["model"])
    results = []
    for trial in worker_trials:
        config = copy.deepcopy(base_config)
        config["selector"] = trial["selector_config"]
        if trial.get("action_space") is not None:
            config["action_space"] = trial["action_space"]
        payload = build_arm_payload(raw_payload, spec, trial["arm"])
        payload = merge_dev_splits(payload, into="dev_tune", merge_from=("dev_select",))
        trial_dir = Path(output_dir) / trial["trial_id"]
        t0 = time.time()
        try:
            result = train_selector(
                train_dev_pt=Path(train_dev_pt), output_dir=trial_dir, config=config,
                device=device, payload=payload,
            )
            result.update({"trial_id": trial["trial_id"], "status": "completed", "device": device})
        except Exception as exc:  # noqa: BLE001
            result = {
                "trial_id": trial["trial_id"], "status": "failed", "device": device,
                "error": str(exc), "traceback": traceback.format_exc(), "elapsed_sec": time.time() - t0,
            }
        trial_dir.mkdir(parents=True, exist_ok=True)
        with (trial_dir / "trial_meta.json").open("w", encoding="utf-8") as f:
            json.dump({**trial, "result": result}, f, indent=2, default=str)
        print(f"[{trial['trial_id']}] {result['status']} on {device} "
              f"(epoch={result.get('best_epoch')}, dmr={result.get('best_dev_tune_dmr')}, "
              f"elapsed={result.get('elapsed_sec', 0):.1f}s)", flush=True)
        results.append(result)
    result_queue.put(results)

def run_trials(trials: list[dict], base_config: dict, train_dev_pt: Path, output_dir: Path,
               gpus: list[str], plan: dict | None = None) -> list[dict]:
    """Train the input-representation trials. The worker is rebound because spawned processes re-import the module.
    """
    import ablation_common as MEC

    MEC.train_worker = train_worker_arm
    return MEC.run_trials(trials, base_config, train_dev_pt, output_dir, gpus, plan)

def build_plan(spec, base_config: dict, arms: list[str], trials: list[dict], meta: dict) -> dict:
    return {
        "schema_version": "selfmorph-input-rep-ablation-plan-1",
        "purpose": "10-layer trajectory, one hidden state, or the prompt embedding; "
                   "only the selector input changes; the released selector configuration is otherwise unchanged",
        "model": spec.key,
        "label": spec.label,
        "config": spec.config,
        "train_dev_pt": spec.train_dev_pt,
        "selector_checkpoint": spec.checkpoint,
        "traj10_reused_checkpoints": str(TRAJ10_CKPT_DIR.relative_to(REPO_ROOT) / spec.key),
        "arms": arms,
        "num_trials": len(trials),
        "arm_geometry": meta,
        "embed_payload_source": str(embed_vectors_path(spec).relative_to(REPO_ROOT)),
        "trials": [{k: v for k, v in t.items() if k not in ("selector_config",)} for t in trials],
    }

if __name__ == "__main__":
    rows = []
    for spec in model_specs():
        cfg = load_config(spec)
        rows.append({
            "model": spec.key, "dense_layer_ids": dense_layer_ids(cfg),
            "single_slot": SINGLE_SLOT, "single_layer": single_layer(cfg),
            "single_gain": broadcast_gain_name("single", cfg, spec.key),
            "embed_gain": broadcast_gain_name("embed", cfg, spec.key),
            "authoritative_gain": str(authoritative_gain_path(cfg).relative_to(REPO_ROOT)),
        })
    print(json.dumps(rows, indent=2))
