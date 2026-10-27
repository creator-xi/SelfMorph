"""Trajectory-conditioned morphology selector (encoder, trunk, and head)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from selfmorph.action_space import Anchor, AnchorSet
from selfmorph.settings import PACKAGE_ROOT

NUM_TYPES = 3
TYPE_BASE, TYPE_REPEAT, TYPE_BRANCH = 0, 1, 2
TYPE_ORDER: tuple[int, ...] = (TYPE_BASE, TYPE_REPEAT, TYPE_BRANCH)
TYPE_NAME_BY_ID: dict[int, str] = {TYPE_BASE: "BASE", TYPE_REPEAT: "REPEAT", TYPE_BRANCH: "BRANCH"}

DEFAULT_PRENORM_GAIN_PATH = PACKAGE_ROOT / "resources" / "base_model_prenorm" / "qwen35_27b_input_layernorm_gain.pt"

@lru_cache(maxsize=4)
def load_prenorm_gain_payload(path: str | Path = DEFAULT_PRENORM_GAIN_PATH) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)

def load_prenorm_gain(path: str | Path = DEFAULT_PRENORM_GAIN_PATH) -> torch.Tensor:
    return load_prenorm_gain_payload(path)["gain"].float()

def prenorm(
    h: torch.Tensor, *, mode: str = "rms_gain", gain: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> torch.Tensor:
    """Normalize the last dimension of `h` (`[..., D]`)."""
    x = h.float()
    if mode == "ln":
        xn = F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=eps if eps is not None else 1e-5)
    elif mode in ("rms", "rms_gain"):
        xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + (eps if eps is not None else 1e-6))
        if mode == "rms_gain":
            if gain is None:
                raise ValueError("prenorm(mode='rms_gain') requires a gain tensor")
            xn = xn * gain.to(device=xn.device, dtype=xn.dtype)
    elif mode == "none":
        xn = x
    else:
        raise ValueError(f"Unknown prenorm mode {mode!r}, expected 'rms_gain'/'none'/'rms'/'ln'")
    return xn.to(h.dtype)

def resolve_prenorm_gain(
    mode: str, num_layers: int, layer_set: str = "dense",
    gain_path: str | Path = DEFAULT_PRENORM_GAIN_PATH,
    expected_layer_ids: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    if mode != "rms_gain":
        return None
    payload = load_prenorm_gain_payload(gain_path)
    full = payload["gain"].float()
    if num_layers != full.shape[0]:
        raise ValueError(
            f"prenorm_mode='rms_gain' requires trajectory layer count ({num_layers}) "
            f"to match stored gain row count ({full.shape[0]})."
        )
    captured = payload.get("captured_layers")
    if expected_layer_ids is not None and captured is not None:
        expected = [int(x) for x in expected_layer_ids]
        captured_list = [int(x) for x in captured]
        if captured_list != expected:
            raise ValueError(
                f"prenorm gain layer-set mismatch: {gain_path!s} has captured_layers={captured_list}, "
                f"expected {expected}."
            )
    return full

class _BaseEncoder(nn.Module):
    encoder_kind: str = "?"
    seq_dim: Optional[int] = None
    flat_dim: int = 0

    def forward(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

class ProjectionEncoder(_BaseEncoder):
    """Shared learned per-layer linear projection `s_l = W_proj PreNorm(H_l)`."""

    encoder_kind = "projection"

    def __init__(
        self, hidden_dim: int = 5120, num_layers: int = 10, proj_dim: int = 192,
        prenorm_mode: str = "rms_gain", prenorm_gain: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.seq_dim = proj_dim
        self.flat_dim = num_layers * proj_dim
        self.prenorm_mode = prenorm_mode
        self.register_buffer("_gain", prenorm_gain if prenorm_gain is not None else torch.empty(0), persistent=False)

    def forward(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        gain_seq = self._gain if self.prenorm_mode == "rms_gain" else None
        H_n = prenorm(H, mode=self.prenorm_mode, gain=gain_seq)
        z = self.proj(H_n.float())
        return {"e_seq": z, "e_flat": z.flatten(1, 2)}

class DeltaEncoder(_BaseEncoder):
    """Shared linear projection `s_l` concatenated with first-order difference `delta_l = s_l - s_{l-1}`."""

    encoder_kind = "delta"

    def __init__(
        self, hidden_dim: int = 3840, num_layers: int = 10, proj_dim: int = 384,
        prenorm_mode: str = "rms_gain", prenorm_gain: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.seq_dim = 2 * proj_dim
        self.flat_dim = num_layers * 2 * proj_dim
        self.prenorm_mode = prenorm_mode
        self.register_buffer("_gain", prenorm_gain if prenorm_gain is not None else torch.empty(0), persistent=False)

    def forward(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        gain_seq = self._gain if self.prenorm_mode == "rms_gain" else None
        H_n = prenorm(H, mode=self.prenorm_mode, gain=gain_seq)
        s = self.proj(H_n.float())
        d = torch.zeros_like(s)
        d[:, 1:] = s[:, 1:] - s[:, :-1]
        e_seq = torch.cat([s, d], dim=-1)
        return {"e_seq": e_seq, "e_flat": e_seq.flatten(1, 2)}

def build_encoder(
    name: str, *, hidden_dim: int = 5120, num_layers: int = 10,
    prenorm_mode: str = "rms_gain", prenorm_gain: Optional[torch.Tensor] = None, **kwargs,
) -> _BaseEncoder:
    if name == "projection":
        return ProjectionEncoder(
            hidden_dim=hidden_dim, num_layers=num_layers,
            prenorm_mode=prenorm_mode, prenorm_gain=prenorm_gain, **kwargs,
        )
    if name == "delta":
        return DeltaEncoder(
            hidden_dim=hidden_dim, num_layers=num_layers,
            prenorm_mode=prenorm_mode, prenorm_gain=prenorm_gain, **kwargs,
        )
    raise ValueError(f"Unknown encoder {name!r}, expected 'projection' or 'delta'")

class AttentionTrunk(nn.Module):
    """Bidirectional pre-norm Transformer block over depth-embedded layer tokens with prepended `[MORPH]` token."""

    input_kind = "seq"

    def __init__(
        self, seq_dim: int, num_layers: int, d_model: int, n_heads: int = 4, n_blocks: int = 1,
        ffn_inside_mult: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        self.fuse = nn.Linear(seq_dim, d_model)
        self.depth_emb = nn.Embedding(num_layers, d_model)
        self.morph_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_inside_mult * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(ffn_inside_mult * d_model, d_model), nn.Dropout(dropout),
                ),
            }) for _ in range(n_blocks)
        ])
        self.output = nn.LayerNorm(d_model)

    def forward(self, e_seq: torch.Tensor) -> torch.Tensor:
        B, L, _ = e_seq.shape
        z = self.fuse(e_seq) + self.depth_emb(torch.arange(L, device=e_seq.device)).unsqueeze(0)
        tok = self.morph_token.expand(B, 1, -1)
        seq = torch.cat([tok, z], dim=1)
        for block in self.blocks:
            q = block["ln1"](seq)
            a, _ = block["attn"](q, q, q, need_weights=False)
            seq = seq + a
            seq = seq + block["ffn"](block["ln2"](seq))
        return self.output(seq[:, 0])

class MeanPoolTrunk(nn.Module):
    """Position-wise FFN and mean pooling, without self-attention."""

    input_kind = "seq"

    def __init__(
        self, seq_dim: int, num_layers: int, d_model: int, n_blocks: int = 1,
        ffn_inside_mult: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        self.fuse = nn.Linear(seq_dim, d_model)
        self.depth_emb = nn.Embedding(num_layers, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_inside_mult * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(ffn_inside_mult * d_model, d_model), nn.Dropout(dropout),
                ),
            }) for _ in range(n_blocks)
        ])
        self.output = nn.LayerNorm(d_model)

    def forward(self, e_seq: torch.Tensor) -> torch.Tensor:
        B, L, _ = e_seq.shape
        z = self.fuse(e_seq) + self.depth_emb(torch.arange(L, device=e_seq.device)).unsqueeze(0)
        for block in self.blocks:
            z = z + block["ffn"](block["ln"](z))
        return self.output(z.mean(dim=1))

def build_trunk(
    name: str, *, encoder_kind: str, seq_dim: Optional[int] = None, num_layers: Optional[int] = None,
    flat_dim: Optional[int] = None, out_dim: int = 384, dropout: float = 0.1, trunk_depth: int = 0,
    delta_fuse_dim: Optional[int] = None, attention_d_model: Optional[int] = None,
    attention_n_heads: int = 4, attention_n_blocks: int = 1,
) -> nn.Module:
    if encoder_kind not in ("delta", "projection"):
        raise ValueError(f"Expected encoder='projection' or 'delta', got {encoder_kind!r}")
    if seq_dim is None or num_layers is None or attention_d_model is None:
        raise ValueError(f"{name} trunk requires seq_dim, num_layers, and attention_d_model")
    if name == "attention":
        return AttentionTrunk(
            seq_dim=seq_dim, num_layers=num_layers, d_model=attention_d_model,
            n_heads=attention_n_heads, n_blocks=attention_n_blocks, dropout=dropout,
        )
    if name == "mean_pool":
        return MeanPoolTrunk(
            seq_dim=seq_dim, num_layers=num_layers, d_model=attention_d_model,
            n_blocks=attention_n_blocks, dropout=dropout,
        )
    raise ValueError(f"Unknown trunk {name!r}, expected 'attention' or 'mean_pool'")

@dataclass(frozen=True)
class HeadGrid:
    rep_spans: tuple[tuple[int, int], ...]
    rep_r_values: tuple[int, ...]
    br_spans: tuple[tuple[int, int], ...]
    br_k_values: tuple[int, ...]

    @property
    def num_rep_spans(self) -> int:
        return len(self.rep_spans)

    @property
    def num_r(self) -> int:
        return len(self.rep_r_values)

    @property
    def num_br_spans(self) -> int:
        return len(self.br_spans)

    @property
    def num_K(self) -> int:
        return len(self.br_k_values)

    def rep_span_index(self, span: tuple[int, int]) -> Optional[int]:
        try:
            return self.rep_spans.index(tuple(span))
        except ValueError:
            return None

    def rep_r_index(self, r: int) -> Optional[int]:
        try:
            return self.rep_r_values.index(int(r))
        except ValueError:
            return None

    def br_span_index(self, span: tuple[int, int]) -> Optional[int]:
        try:
            return self.br_spans.index(tuple(span))
        except ValueError:
            return None

    def br_k_index(self, K: int) -> Optional[int]:
        try:
            return self.br_k_values.index(int(K))
        except ValueError:
            return None

def head_grid_from_anchors(anchors: AnchorSet) -> HeadGrid:
    rep_spans = sorted({a.layer_span for a in anchors if a.type == "REPEAT"})
    rep_r = sorted({a.r for a in anchors if a.type == "REPEAT"})
    br_spans = sorted({a.layer_span for a in anchors if a.type == "BRANCH"})
    br_k = sorted({a.K for a in anchors if a.type == "BRANCH"})
    return HeadGrid(
        rep_spans=tuple(rep_spans), rep_r_values=tuple(rep_r),
        br_spans=tuple(br_spans), br_k_values=tuple(br_k),
    )

def anchor_type_id(anchor: Anchor) -> int:
    return {"BASE": TYPE_BASE, "REPEAT": TYPE_REPEAT, "BRANCH": TYPE_BRANCH}[anchor.type]

def active_types_from_anchors(anchors: AnchorSet) -> tuple[int, ...]:
    present = {anchor_type_id(a) for a in anchors}
    return tuple(t for t in TYPE_ORDER if t in present)

def active_types_from_grid(grid: HeadGrid) -> tuple[int, ...]:
    active = [TYPE_BASE]
    if grid.num_rep_spans > 0 and grid.num_r > 0:
        active.append(TYPE_REPEAT)
    if grid.num_br_spans > 0 and grid.num_K > 0:
        active.append(TYPE_BRANCH)
    return tuple(active)

@dataclass
class MorphHeadConfig:
    in_dim: int
    grid: HeadGrid
    dropout: float = 0.1
    variant: str = "hybrid"
    head_hidden_dim: int = 128
    active_types: Optional[tuple[int, ...]] = None

class MorphHead(nn.Module):
    """Hierarchical prediction head: `p(m|H) = p(tau|H) * p(span, theta | tau, H)`."""

    def __init__(self, cfg: MorphHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid
        H = cfg.in_dim
        S_rep, R = self.grid.num_rep_spans, self.grid.num_r
        S_br, K = self.grid.num_br_spans, self.grid.num_K

        self.variant = cfg.variant.lower()
        if self.variant != "hybrid":
            raise ValueError(f"Unknown head variant {cfg.variant!r}")

        active: tuple[int, ...] = (
            tuple(int(t) for t in cfg.active_types) if cfg.active_types is not None
            else active_types_from_grid(self.grid)
        )
        if not active:
            raise ValueError("MorphHead requires at least one active action type (BASE)")
        if active != tuple(t for t in TYPE_ORDER if t in set(active)):
            raise ValueError(f"active_types must be a subsequence of {TYPE_ORDER} in that order, got {active}")
        self.active_types: tuple[int, ...] = active
        self.type_col: dict[int, int] = {t: i for i, t in enumerate(active)}
        self.has_type_rep = TYPE_REPEAT in self.type_col
        self.has_type_br = TYPE_BRANCH in self.type_col
        if TYPE_BASE not in self.type_col:
            raise ValueError("BASE must be active: every pool contains the base (BASE) action")
        if self.has_type_rep and (S_rep <= 0 or R <= 0):
            raise ValueError("REPEAT is active but the head grid has no REPEAT span/intensity")
        if self.has_type_br and (S_br <= 0 or K <= 0):
            raise ValueError("BRANCH is active but the head grid has no BRANCH span/intensity")

        self.head_type = nn.Linear(H, len(self.active_types))
        h_hid = cfg.head_hidden_dim
        if self.has_type_rep:
            self.head_sub_REP = nn.Sequential(
                nn.Linear(H, h_hid), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h_hid, S_rep * R),
            )
        if self.has_type_br:
            self.head_sub_BR = nn.Sequential(
                nn.Linear(H, h_hid), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h_hid, S_br * K),
            )

        keys = list(self.state_dict().keys())
        for family, flag in (("REPEAT", self.has_type_rep), ("BR", self.has_type_br)):
            if not flag:
                leaked = [k for k in keys if family in k]
                if leaked:
                    raise RuntimeError(f"head built for a pool without {family} still exposes {leaked}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        S_rep, R = self.grid.num_rep_spans, self.grid.num_r
        S_br, K = self.grid.num_br_spans, self.grid.num_K
        out: dict[str, torch.Tensor] = {"type_logits": self.head_type(x)}
        if self.has_type_rep:
            out["sub_REP_logits"] = self.head_sub_REP(x).view(-1, S_rep, R)
        if self.has_type_br:
            out["sub_BR_logits"] = self.head_sub_BR(x).view(-1, S_br, K)
        return out

    def log_probs_factored(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logs: dict[str, torch.Tensor] = {"log_type": F.log_softmax(out["type_logits"], dim=-1)}
        if self.has_type_rep:
            sub_REP = out["sub_REP_logits"]
            B, S_rep, R = sub_REP.shape
            log_joint_REP = F.log_softmax(sub_REP.view(B, S_rep * R), dim=-1).view(B, S_rep, R)
            log_span_REP = torch.logsumexp(log_joint_REP, dim=-1)
            logs["log_span_REP"] = log_span_REP
            logs["log_r"] = log_joint_REP - log_span_REP.unsqueeze(-1)
        if self.has_type_br:
            sub_BR = out["sub_BR_logits"]
            B, S_br, K = sub_BR.shape
            log_joint_BR = F.log_softmax(sub_BR.view(B, S_br * K), dim=-1).view(B, S_br, K)
            log_span_BR = torch.logsumexp(log_joint_BR, dim=-1)
            logs["log_span_BR"] = log_span_BR
            logs["log_K"] = log_joint_BR - log_span_BR.unsqueeze(-1)
        return logs

    def joint_logp_anchors(self, out: dict[str, torch.Tensor], anchors: AnchorSet) -> torch.Tensor:
        logs = self.log_probs_factored(out)
        B = logs["log_type"].shape[0]
        joint = torch.empty(B, len(anchors), device=logs["log_type"].device, dtype=logs["log_type"].dtype)
        for a in anchors:
            if a.type == "BASE":
                joint[:, a.id] = logs["log_type"][:, self.type_col[TYPE_BASE]]
            elif a.type == "REPEAT":
                si, ri = self.grid.rep_span_index(a.layer_span), self.grid.rep_r_index(a.r)
                if si is None or ri is None:
                    raise ValueError(f"Anchor {a.id} (REPEAT span={a.layer_span} r={a.r}) not in head grid")
                joint[:, a.id] = logs["log_type"][:, self.type_col[TYPE_REPEAT]] + logs["log_span_REP"][:, si] + logs["log_r"][:, si, ri]
            else:
                si, ki = self.grid.br_span_index(a.layer_span), self.grid.br_k_index(a.K)
                if si is None or ki is None:
                    raise ValueError(f"Anchor {a.id} (BR span={a.layer_span} K={a.K}) not in head grid")
                joint[:, a.id] = logs["log_type"][:, self.type_col[TYPE_BRANCH]] + logs["log_span_BR"][:, si] + logs["log_K"][:, si, ki]
        return joint

def selector_loss(
    out: dict[str, torch.Tensor], p_star: torch.Tensor, head: MorphHead, anchors: AnchorSet,
    *, weight_type: float = 1.0, weight_span: float = 1.0, weight_param: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Cross-entropy against the utility distribution."""
    grid = head.grid
    B = p_star.shape[0]
    device, dtype = p_star.device, p_star.dtype
    logs = head.log_probs_factored(out)

    p_type = torch.zeros(B, len(head.active_types), device=device, dtype=dtype)
    if head.has_type_rep:
        p_span_REP = torch.zeros(B, grid.num_rep_spans, device=device, dtype=dtype)
    if head.has_type_br:
        p_span_BR = torch.zeros(B, grid.num_br_spans, device=device, dtype=dtype)
    if head.has_type_rep:
        p_r_joint = torch.zeros(B, grid.num_rep_spans, grid.num_r, device=device, dtype=dtype)
    if head.has_type_br:
        p_K_joint = torch.zeros(B, grid.num_br_spans, grid.num_K, device=device, dtype=dtype)

    for a in anchors:
        t = head.type_col[anchor_type_id(a)]
        p_type[:, t] += p_star[:, a.id]
        if a.type == "REPEAT":
            si, ri = grid.rep_span_index(a.layer_span), grid.rep_r_index(a.r)
            p_span_REP[:, si] += p_star[:, a.id]
            p_r_joint[:, si, ri] += p_star[:, a.id]
        elif a.type == "BRANCH":
            si, ki = grid.br_span_index(a.layer_span), grid.br_k_index(a.K)
            p_span_BR[:, si] += p_star[:, a.id]
            p_K_joint[:, si, ki] += p_star[:, a.id]

    L_type = -(p_type * logs["log_type"]).sum(dim=-1).mean()

    L_span_total: Optional[torch.Tensor] = None
    L_param_total: Optional[torch.Tensor] = None
    L_span_REP = L_span_BR = L_r = L_K = None

    if head.has_type_rep:
        weight_REP = p_type[:, head.type_col[TYPE_REPEAT]]
        p_span_REP_cond = p_span_REP / (weight_REP.unsqueeze(-1) + eps)
        L_span_REP = -(weight_REP * (p_span_REP_cond * logs["log_span_REP"]).sum(-1)).mean()
        L_r = -(p_r_joint * logs["log_r"]).sum((-1, -2)).mean()
        L_span_total = L_span_REP
        L_param_total = L_r
    if head.has_type_br:
        weight_BR = p_type[:, head.type_col[TYPE_BRANCH]]
        p_span_BR_cond = p_span_BR / (weight_BR.unsqueeze(-1) + eps)
        L_span_BR = -(weight_BR * (p_span_BR_cond * logs["log_span_BR"]).sum(-1)).mean()
        L_K = -(p_K_joint * logs["log_K"]).sum((-1, -2)).mean()
        L_span_total = L_span_BR if L_span_total is None else L_span_total + L_span_BR
        L_param_total = L_K if L_param_total is None else L_param_total + L_K

    L_total = weight_type * L_type + weight_span * L_span_total + weight_param * L_param_total
    result: dict[str, torch.Tensor] = {"loss": L_total, "L_type": L_type.detach()}
    for name, term in (("L_span_REP", L_span_REP), ("L_span_BR", L_span_BR), ("L_r", L_r), ("L_K", L_K)):
        if term is not None:
            result[name] = term.detach()
    return result

class MorphologySelector(nn.Module):
    """Full trajectory-conditioned selector (`Encoder -> Trunk -> Head`)."""

    def __init__(self, encoder: _BaseEncoder, trunk: nn.Module, head: MorphHead):
        super().__init__()
        self.encoder = encoder
        self.trunk = trunk
        self.head = head

    def forward(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(H)
        if getattr(self.trunk, "input_kind", "flat") == "seq":
            z = self.trunk(feats["e_seq"])
        else:
            z = self.trunk(feats["e_flat"])
        return self.head(z)

def build_selector(
    selector_cfg: dict, anchors: AnchorSet, hidden_size: int, num_traj_layers: int,
) -> MorphologySelector:
    prenorm_mode = selector_cfg.get("prenorm_mode", "rms_gain")
    gain_path = selector_cfg.get("prenorm_gain_path")
    if gain_path:
        gain_path = Path(gain_path)
        if not gain_path.is_absolute():
            gain_path = PACKAGE_ROOT / gain_path
    else:
        gain_path = DEFAULT_PRENORM_GAIN_PATH
    prenorm_gain = resolve_prenorm_gain(
        prenorm_mode, num_traj_layers, layer_set=selector_cfg.get("layer_set", "dense"),
        gain_path=gain_path, expected_layer_ids=selector_cfg.get("expected_trajectory_layer_ids"),
    )

    encoder = build_encoder(
        selector_cfg["encoder"], hidden_dim=hidden_size, num_layers=num_traj_layers,
        proj_dim=selector_cfg["proj_dim"], prenorm_mode=prenorm_mode, prenorm_gain=prenorm_gain,
    )
    trunk = build_trunk(
        selector_cfg["trunk"], encoder_kind=encoder.encoder_kind, seq_dim=encoder.seq_dim,
        num_layers=num_traj_layers, flat_dim=encoder.flat_dim, out_dim=selector_cfg["trunk_hidden"],
        dropout=selector_cfg["dropout"], trunk_depth=selector_cfg.get("trunk_depth", 0),
        delta_fuse_dim=selector_cfg.get("proj_dim"), attention_d_model=selector_cfg.get("attention_d_model"),
        attention_n_heads=selector_cfg.get("attention_n_heads", 4),
        attention_n_blocks=selector_cfg.get("attention_n_blocks", 1),
    )
    head_in_dim = (
        selector_cfg.get("attention_d_model")
        if selector_cfg["trunk"] in ("attention", "mean_pool")
        else selector_cfg["trunk_hidden"]
    )
    grid = head_grid_from_anchors(anchors)
    head = MorphHead(MorphHeadConfig(
        in_dim=head_in_dim, grid=grid, dropout=selector_cfg["dropout"],
        variant=selector_cfg["head_variant"], head_hidden_dim=selector_cfg["head_hidden_dim"],
        active_types=active_types_from_anchors(anchors),
    ))
    return MorphologySelector(encoder, trunk, head)
