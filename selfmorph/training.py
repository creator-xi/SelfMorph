"""Utility-distillation training for the morphology selector."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from selfmorph.action_space import AnchorSet, build_action_space
from selfmorph.selector import MorphologySelector, build_selector, selector_loss
from selfmorph.settings import load_runtime_config

@dataclass
class TemperatureSchedule:
    """Distillation temperature `T(epoch)`: fixed or cosine-annealed."""

    mode: str = "fixed"
    T_fixed: float = 0.15
    T_max: float = 0.15
    T_min: float = 0.15
    num_epochs: int = 50

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "cosine"):
            raise ValueError(f"Unknown temperature mode {self.mode!r}")
        if self.mode == "fixed" and self.T_fixed <= 0:
            raise ValueError("T_fixed must be > 0")
        if self.mode == "cosine" and not (0 < self.T_min <= self.T_max):
            raise ValueError("cosine schedule requires 0 < T_min <= T_max")

    def T(self, epoch: int) -> float:
        if self.mode == "fixed":
            return float(self.T_fixed)
        e = max(0, min(int(epoch), int(self.num_epochs)))
        frac = math.pi * e / float(self.num_epochs)
        return float(self.T_min + 0.5 * (self.T_max - self.T_min) * (1.0 + math.cos(frac)))

@dataclass
class LRSchedule:
    """Cosine-decay-with-linear-warmup learning rate schedule (`scale(step)` in `[min_scale, 1.0]`)."""

    mode: str = "cosine_warmup"
    total_steps: int = 1
    warmup_frac: float = 0.05
    min_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.mode not in ("none", "cosine_warmup"):
            raise ValueError(f"Unknown lr_schedule mode {self.mode!r}")
        self.warmup_steps = max(1, int(self.warmup_frac * max(1, self.total_steps)))

    def scale(self, step: int) -> float:
        if self.mode == "none":
            return 1.0
        if step <= self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(self.min_scale, cosine)

def build_optimizer(selector: torch.nn.Module, s1: dict) -> torch.optim.Optimizer:
    lr = s1["lr"]
    wd = s1["weight_decay"]
    if not s1.get("weight_decay_exempt_norm_bias", True):
        return torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=wd)
    decay, no_decay = [], []
    for name, p in selector.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr,
    )

class TrajectoryDataset(Dataset):
    """Wraps `train_dev.pt` tensors for one split."""

    def __init__(self, payload: dict, split: str):
        mask = [s == split for s in payload["splits"]]
        idx = torch.tensor([i for i, m in enumerate(mask) if m], dtype=torch.long)
        if len(idx) == 0:
            raise ValueError(f"No records for split={split!r} in train_dev.pt")
        self.trajectory = payload["trajectory"][idx]
        self.utilities = payload["utilities"][idx]
        self.rewards = payload["rewards"][idx]
        self.scores = payload["scores"][idx]
        self.domains = [payload["domains"][i] for i in idx.tolist()]

    def __len__(self) -> int:
        return self.trajectory.shape[0]

    def __getitem__(self, i: int):
        return self.trajectory[i], self.utilities[i], self.rewards[i], self.domains[i]

def domain_macro_metrics(
    selector: MorphologySelector, anchors: AnchorSet, dataset: TrajectoryDataset, device: str,
    batch_size: int = 256,
) -> dict[str, float]:
    """Domain-macro-averaged utility (`domain_macro_reward`) and score under the selector's argmax policy."""
    selector.eval()
    n = len(dataset)
    all_reward = torch.empty(n)
    all_score = torch.empty(n)
    all_domain: list[str] = []
    traj_device = dataset.trajectory.device
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            traj = (
                dataset.trajectory[start:end]
                if traj_device.type == "cuda"
                else dataset.trajectory[start:end].to(device, non_blocking=True)
            )
            out = selector(traj)
            logp = selector.head.joint_logp_anchors(out, anchors)
            chosen = logp.argmax(dim=-1).cpu()
            rows = torch.arange(end - start)
            all_reward[start:end] = dataset.rewards[start:end][rows, chosen]
            all_score[start:end] = dataset.scores[start:end][rows, chosen]
            all_domain.extend(dataset.domains[start:end])

    per_domain_reward: dict[str, list[float]] = {}
    per_domain_score: dict[str, list[float]] = {}
    for i, domain in enumerate(all_domain):
        per_domain_reward.setdefault(domain, []).append(float(all_reward[i]))
        per_domain_score.setdefault(domain, []).append(float(all_score[i]))
    if not per_domain_reward:
        return {"domain_macro_reward": 0.0, "domain_macro_score": 0.0}
    dmr = sum(sum(v) / len(v) for v in per_domain_reward.values()) / len(per_domain_reward)
    dms = sum(sum(v) / len(v) for v in per_domain_score.values()) / len(per_domain_score)
    return {"domain_macro_reward": dmr, "domain_macro_score": dms}

def train_selector(
    *,
    train_dev_pt: Path,
    output_dir: Path,
    config: dict | None = None,
    device: str = "cpu",
    payload: dict | None = None,
) -> dict:
    config = config or load_runtime_config()
    s1 = config["selector"]
    torch.manual_seed(s1["seed"])

    if payload is None:
        payload = torch.load(train_dev_pt, map_location="cpu", weights_only=False)
    anchors = build_action_space(config)
    if payload["num_anchors"] != len(anchors):
        raise ValueError(
            f"train_dev.pt was built with {payload['num_anchors']} anchors, "
            f"but config enumerates {len(anchors)}."
        )

    train_ds = TrajectoryDataset(payload, "train")
    dev_tune_ds = TrajectoryDataset(payload, "dev_tune")
    if device.startswith("cuda"):
        train_ds.trajectory = train_ds.trajectory.to(device)
        train_ds.utilities = train_ds.utilities.to(device)
        dev_tune_ds.trajectory = dev_tune_ds.trajectory.to(device)

    hidden_size = config["model"]["hidden_size"]
    num_traj_layers = train_ds.trajectory.shape[1]

    selector = build_selector(s1, anchors, hidden_size, num_traj_layers).to(device)
    optimizer = build_optimizer(selector, s1)
    temp_sched = TemperatureSchedule(
        mode=s1["temperature_mode"], T_fixed=s1["temperature_fixed"],
        T_max=s1["temperature_cosine_start"], T_min=s1["temperature_cosine_end"],
        num_epochs=s1.get("schedule_epochs", s1["max_epochs"]),
    )

    effective_batch = int(s1.get("effective_batch_size", s1["batch_size"]))
    micro_batch = int(s1.get("micro_batch_size", effective_batch))
    if effective_batch % micro_batch != 0:
        raise ValueError("selector.effective_batch_size must be divisible by selector.micro_batch_size")
    n_train = len(train_ds)

    steps_per_epoch = math.ceil(n_train / effective_batch)
    total_steps = steps_per_epoch * s1["max_epochs"]
    lr_sched = LRSchedule(
        mode=s1.get("lr_schedule", "cosine_warmup"), total_steps=total_steps,
        warmup_frac=s1.get("lr_warmup_frac", 0.05), min_scale=s1.get("lr_min_scale", 0.02),
    )
    global_step = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best_checkpoint.pt"
    best_metrics: dict[str, float] | None = None
    best_epoch = -1
    patience = s1["patience"]
    epochs_since_best = 0
    start_time = time.time()

    order_device = train_ds.trajectory.device

    with history_path.open("w", encoding="utf-8") as history_f:
        epoch = -1
        for epoch in range(s1["max_epochs"]):
            selector.train()
            T = temp_sched.T(epoch)
            loss_sum = torch.zeros((), device=order_device)
            n_samples = 0
            order = torch.randperm(n_train, device=order_device)
            for eff_start in range(0, n_train, effective_batch):
                eff_idx = order[eff_start:eff_start + effective_batch]
                n_in_batch = eff_idx.numel()
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, n_in_batch, micro_batch):
                    micro_idx = eff_idx[micro_start:micro_start + micro_batch]
                    micro_traj = (
                        train_ds.trajectory[micro_idx]
                        if order_device.type == "cuda"
                        else train_ds.trajectory[micro_idx].to(device, non_blocking=True)
                    )
                    micro_util = (
                        train_ds.utilities[micro_idx]
                        if order_device.type == "cuda"
                        else train_ds.utilities[micro_idx].to(device, non_blocking=True)
                    )
                    p_star = torch.softmax(micro_util / T, dim=-1)
                    out = selector(micro_traj)
                    loss_dict = selector_loss(out, p_star, selector.head, anchors)
                    n_micro = micro_idx.numel()
                    (loss_dict["loss"] * (n_micro / n_in_batch)).backward()
                    loss_sum += loss_dict["loss"].detach() * n_micro
                    n_samples += n_micro
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
                global_step += 1
                lr_scale = lr_sched.scale(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = s1["lr"] * lr_scale
                optimizer.step()

            metrics = domain_macro_metrics(selector, anchors, dev_tune_ds, device)
            record = {
                "epoch": epoch, "temperature": T, "train_loss": float(loss_sum.item()) / max(n_samples, 1),
                "dev_tune_dmr": metrics["domain_macro_reward"], "dev_tune_dms": metrics["domain_macro_score"],
            }
            history_f.write(json.dumps(record) + "\n")
            history_f.flush()

            improved = best_metrics is None or (
                metrics["domain_macro_reward"], metrics["domain_macro_score"]
            ) > (
                best_metrics["domain_macro_reward"] + 1e-5, best_metrics["domain_macro_score"]
            )
            if improved:
                best_metrics = metrics
                best_epoch = epoch
                epochs_since_best = 0
                torch.save({
                    "model_state": selector.state_dict(), "epoch": epoch,
                    "dev_tune_dmr": metrics["domain_macro_reward"], "dev_tune_dms": metrics["domain_macro_score"],
                }, best_path)
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    break

    if best_metrics is None:
        raise RuntimeError("training completed without saving a checkpoint")

    return {
        "best_epoch": best_epoch,
        "best_dev_tune_dmr": best_metrics["domain_macro_reward"],
        "best_dev_tune_dms": best_metrics["domain_macro_score"],
        "checkpoint_selection": "best",
        "epochs_run": epoch + 1,
        "elapsed_sec": time.time() - start_time,
        "checkpoint_path": str(best_path),
    }
