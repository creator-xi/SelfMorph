
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from selfmorph.action_space import build_action_space, repeat_layer_indices  # noqa: E402
from selfmorph.benchmarks import model_wants_qwen_think_seed, render_prompt  # noqa: E402
from selfmorph.core.branch_model import _replicate_inputs, build_branch_model  # noqa: E402
from selfmorph.core.layer_duplicator import build_model_with_layers  # noqa: E402
from selfmorph.executor import _build_branch_config  # noqa: E402
from selfmorph.model import ModelConfig, get_decoder_layers, load_frozen_model  # noqa: E402
from selfmorph.oracle import _manifest_samples  # noqa: E402
from selfmorph.trajectory import collect_trajectory, stack_trajectory  # noqa: E402

class ExecOrderCapture:
    """Last-token hidden state at every EXECUTED decoder step, in execution
    order (see this module's docstring for why hooks and not
    `output_hidden_states`). Read-only: never modifies the hidden state.
    """

    def __init__(self, model):
        self._layers = list(get_decoder_layers(model))
        self.captured: list[torch.Tensor] = []
        self.embedding: torch.Tensor | None = None
        self._handles: list = []

    def _post(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.captured.append(h[:, -1, :].detach().to(torch.float32).cpu())

    def _pre(self, module, inputs):
        if self.embedding is None and inputs and isinstance(inputs[0], torch.Tensor):
            self.embedding = inputs[0][:, -1, :].detach().to(torch.float32).cpu()

    def __enter__(self) -> "ExecOrderCapture":
        for layer in self._layers:
            self._handles.append(layer.register_forward_hook(self._post))
        self._handles.append(self._layers[0].register_forward_pre_hook(self._pre))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def states(self) -> torch.Tensor:
        """[T+1, rows, hidden]: embedding output followed by every executed
        step's raw output.
        """
        if self.embedding is None:
            raise RuntimeError("layer-0 pre-hook never fired")
        return torch.stack([self.embedding] + self.captured, dim=0)

@torch.no_grad()
def capture_base(model, enc) -> torch.Tensor:
    with ExecOrderCapture(model) as cap:
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    return cap.states()

@torch.no_grad()
def capture_rep(model, enc, anchor, num_layers) -> tuple[torch.Tensor, list[int]]:
    layer_indices = repeat_layer_indices(num_layers, anchor.layer_span, anchor.r)
    custom = build_model_with_layers(model, layer_indices)
    try:
        with ExecOrderCapture(model) as cap:
            custom(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                   use_cache=False)
        return cap.states(), layer_indices
    finally:
        del custom

@torch.no_grad()
def capture_br(model, enc, anchor, action_cfg) -> torch.Tensor:
    """Replicate the batch K-fold and run one prefill inside the production
    SplitMergeModel hook context, so the perturbation seed/schedule and the
    merge point are exactly those of `executor.execute_anchor`.

    The capture context is entered BEFORE `_activate` so that the layer-0
    pre-hook reads the un-perturbed embedding even when branch_start == 0.
    """
    cfg = _build_branch_config(anchor, action_cfg, action_cfg["branch_sigma_max"])
    wrapper = build_branch_model(model, cfg)
    try:
        B = int(enc["input_ids"].shape[0])
        rep_ids, rep_mask = _replicate_inputs(enc["input_ids"], enc["attention_mask"], anchor.K)
        with ExecOrderCapture(model) as cap:
            with wrapper._activate(batch_size=B):
                model(input_ids=rep_ids, attention_mask=rep_mask, use_cache=False)
        return cap.states()
    finally:
        del wrapper

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "runtime.json"))
    ap.add_argument("--cases", default=str(REPO_ROOT / "artifacts" / "results" / "trajectory" / "cases.json"))
    ap.add_argument("--train-dev-pt", default=str(
        REPO_ROOT / "artifacts" / "training_data" / "qwen35_27b" / "train_dev.pt"),
        help="compare the probed trajectory with the stored training tensor")
    ap.add_argument("--anchors", default="0,4,5,6,10,13,14,15,16,22",
                    help="comma-separated anchor ids to capture per context")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "results" / "trajectory"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    anchors = build_action_space(config)
    by_id = anchors.by_id
    anchor_ids = [int(x) for x in args.anchors.split(",") if x.strip()]
    num_layers = config["model"]["num_layers"]
    action_cfg = config["action_space"]
    dense_layer_ids = list(config["trajectory"]["dense_layer_ids"])

    cases_doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = cases_doc["cases"]
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "traces_index.jsonl"

    done: set[tuple[str, int]] = set()
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["context_id"], int(rec["anchor_id"])))
            except Exception:
                continue
    print(f"[resume] {len(done)} (context, anchor) traces already indexed")

    print(f"[load] model on {args.device} ...", flush=True)
    mcfg = ModelConfig(
        model_path=os.environ["SELFMORPH_MODEL_PATH"],
        num_layers=num_layers,
        hidden_size=config["model"]["hidden_size"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    model, tokenizer = load_frozen_model(device=args.device, model_config=mcfg)
    append_think_seed = model_wants_qwen_think_seed(model)
    print(f"[load] done. append_think_seed={append_think_seed}", flush=True)

    payload = torch.load(args.train_dev_pt, map_location="cpu", weights_only=True)
    stored_traj = payload["trajectory"]

    sample_cache: dict[str, dict] = {}
    gates: list[dict] = []
    index_f = open(index_path, "a", buffering=1, encoding="utf-8")
    n_new = 0
    try:
        for case in cases:
            ds, qid = case["context_id"].split("::", 1)
            if ds not in sample_cache:
                sample_cache[ds] = {s.qid: s for s in _manifest_samples(ds)}
            sample = sample_cache[ds].get(qid)
            if sample is None:
                raise KeyError(f"qid {qid!r} not found in manifest samples for dataset {ds!r}")

            prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            n_tok = int(enc["input_ids"].shape[1])
            ctx_dir = trace_dir / case["context_id"]
            ctx_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{case['group']}] {case['context_id']}  prompt_tokens={n_tok}", flush=True)

            base_states: torch.Tensor | None = None
            for aid in anchor_ids:
                anchor = by_id[aid]
                trace_path = ctx_dir / f"anchor{aid}.pt"
                if (case["context_id"], aid) in done and trace_path.is_file():
                    blob = torch.load(trace_path, map_location="cpu", weights_only=True)
                    if aid == 0:
                        base_states = blob["states"].squeeze(1)
                    print(f"    a{aid:<2d} {anchor.type:9s} cached")
                    continue

                if anchor.type == "BASE":
                    states = capture_base(model, enc)
                    exec_idx = list(range(num_layers))
                elif anchor.type == "REPEAT":
                    states, exec_idx = capture_rep(model, enc, anchor, num_layers)
                else:
                    states = capture_br(model, enc, anchor, action_cfg)
                    exec_idx = list(range(num_layers))

                if states.shape[0] != len(exec_idx) + 1:
                    raise RuntimeError(
                        f"captured {states.shape[0]} states but executed {len(exec_idx)} layers "
                        f"(anchor {aid}, {anchor.type}); capture route is broken")

                blob = {
                    "states": states,
                    "context_id": case["context_id"],
                    "anchor_id": aid,
                    "anchor_type": anchor.type,
                    "layer_span": list(anchor.layer_span) if anchor.layer_span else None,
                    "r": anchor.r, "K": anchor.K,
                    "analytic_cost": anchor.analytic_cost,
                    "exec_layer_indices": exec_idx,
                    "num_layers": num_layers,
                    "capture_route": "exec_order_forward_hooks",
                    "states_index_convention": "states[0]=embedding, states[l+1]=raw output of executed step l",
                    "prompt_tokens": n_tok,
                }
                torch.save(blob, trace_path)
                rec = {
                    "context_id": case["context_id"], "anchor_id": aid, "anchor_type": anchor.type,
                    "group": case["group"], "dataset": case["dataset"],
                    "trace_path": str(trace_path.relative_to(out_dir)),
                    "n_states": int(states.shape[0]), "n_exec_layers": len(exec_idx),
                    "n_rows": int(states.shape[1]), "prompt_tokens": n_tok,
                }
                index_f.write(json.dumps(rec) + "\n")
                index_f.flush()
                os.fsync(index_f.fileno())
                n_new += 1
                print(f"    a{aid:<2d} {anchor.type:9s} states={tuple(states.shape)} exec_layers={len(exec_idx)}")

                if aid == 0:
                    base_states = states.squeeze(1)

            if base_states is None:
                continue
            g = {"context_id": case["context_id"]}

            prod = stack_trajectory(
                collect_trajectory(model, enc, layer_ids=tuple(dense_layer_ids)),
                tuple(dense_layer_ids),
            ).to(torch.float32)
            got = torch.stack([base_states[l + 1] for l in dense_layer_ids], dim=0)
            g["probe_matches"] = bool(torch.equal(got, prod))
            g["probe_max_rel_err"] = float(
                ((got - prod).norm(dim=-1) / prod.norm(dim=-1).clamp_min(1e-9)).max())

            ref = stored_traj[case["row_index"]]
            rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-9))
            g["stored_max_rel_err"] = float(rel.max())
            g["stored_ok"] = bool(rel.max() < 2e-2)

            for aid in anchor_ids:
                anchor = by_id[aid]
                tp = ctx_dir / f"anchor{aid}.pt"
                if not tp.is_file():
                    continue
                blob = torch.load(tp, map_location="cpu", weights_only=True)
                st = blob["states"]
                g[f"depth_a{aid}_states_eq_exec_plus1"] = bool(
                    st.shape[0] == len(blob["exec_layer_indices"]) + 1)
                if anchor.type == "BRANCH":
                    i, j = anchor.layer_span
                    sp_i = float((st[i] - st[i].mean(0, keepdim=True)).norm(dim=-1).max())
                    sp_j1 = float((st[j + 1] - st[j + 1].mean(0, keepdim=True)).norm(dim=-1).max())
                    g[f"branch_entry_a{aid}_spread_at_split_input_is_zero"] = sp_i == 0.0
                    g[f"branch_entry_a{aid}_spread_after_merge_is_zero"] = sp_j1 == 0.0
                    g[f"branch_entry_a{aid}_spread_premerge_rel"] = round(
                        float((st[j] - st[j].mean(0, keepdim=True)).norm(dim=-1).mean()
                              / st[j].norm(dim=-1).mean()), 6)
                    relb = ((st[:, 0, :] - base_states).norm(dim=-1)
                            / base_states.norm(dim=-1).clamp_min(1e-9))
                    pre, post = float(relb[: j + 1].max()), float(relb[j + 1])
                    g[f"branch_vs_base_a{aid}_branch0_vs_base_premerge_max_rel"] = round(pre, 6)
                    g[f"branch_vs_base_a{aid}_postmerge_rel"] = round(post, 6)
                    g[f"branch_vs_base_a{aid}_pass"] = bool(pre < 5e-2 and post > 10 * max(pre, 1e-9))
                    g[f"branch_norm_a{aid}_merge_norm_shrink_ratio"] = round(
                        float(st[j].mean(0).norm() / st[j].norm(dim=-1).mean()), 6)
                elif anchor.type == "REPEAT":
                    _, jj = anchor.layer_span
                    g[f"repeat_prefix_a{aid}_prefix_eq_base_upto_{jj}"] = bool(
                        torch.equal(st[: jj + 1, 0, :], base_states[: jj + 1]))
            gates.append(g)
            print("    gates:", json.dumps({k: v for k, v in g.items() if k != "context_id"}))
    finally:
        os.fsync(index_f.fileno())
        index_f.close()

    (out_dir / "trace_checks.json").write_text(json.dumps({
        "n_new_traces": n_new,
        "anchors": anchor_ids,
        "dense_layer_ids": dense_layer_ids,
        "checks": gates,
    }, indent=2), encoding="utf-8")

    def all_pass(g, prefix, suffix=""):
        vals = [v for k, v in g.items() if k.startswith(prefix) and k.endswith(suffix)]
        return bool(vals) and all(vals)

    n0 = sum(1 for g in gates if g.get("probe_matches"))
    n1 = sum(1 for g in gates if g.get("stored_ok"))
    n2 = sum(1 for g in gates if all_pass(g, "branch_entry_", "_is_zero") and all_pass(g, "branch_vs_base_", "_pass"))
    n3 = sum(1 for g in gates if all_pass(g, "repeat_prefix_"))
    n4 = sum(1 for g in gates if all_pass(g, "depth_"))
    print(f"\n[done] new traces={n_new}; checks over {len(gates)} contexts: "
          f"probe {n0}/{len(gates)}, stored {n1}/{len(gates)}, branch {n2}/{len(gates)}, "
          f"repeat {n3}/{len(gates)}, depth {n4}/{len(gates)}")
    print(f"traces -> {trace_dir}")
    print(f"checks -> {out_dir / 'trace_checks.json'}")

if __name__ == "__main__":
    main()
