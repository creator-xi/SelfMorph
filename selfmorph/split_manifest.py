from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from selfmorph.settings import PACKAGE_ROOT

_MANIFEST_DIR = PACKAGE_ROOT / "resources" / "split_manifests"
_ID_FILE = "id_manifest.json"


@lru_cache(maxsize=1)
def load_split_manifest() -> dict[tuple[str, str, str], dict]:
    """Return ``{(dataset, source_split, loader_qid): {"split": str, "max_new": int | None}}``."""
    with (_MANIFEST_DIR / _ID_FILE).open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[tuple[str, str, str], dict] = {}
    for split_name, rows in data["splits"].items():
        for r in rows:
            out[(r["dataset"], r.get("source_split", "test"), r["loader_qid"])] = {
                "split": split_name,
                "max_new": r.get("max_new"),
            }
    return out


def dataset_source_splits(dataset: str) -> set[str]:
    """Distinct upstream HF splits required for *dataset*."""
    return {src for (ds, src, _) in load_split_manifest() if ds == dataset}


def lookup_split(dataset: str, source_split: str, loader_qid: str) -> str | None:
    """Return train / dev_tune / dev_select / eval, or ``None``."""
    return load_split_manifest().get((dataset, source_split, loader_qid), {}).get("split")


def manifest_datasets() -> set[str]:
    """All in-distribution dataset keys present in the manifest."""
    return {ds for (ds, _, _) in load_split_manifest()}


def split_counts() -> dict[str, dict[str, int]]:
    """Per-dataset per-split row counts."""
    counts: dict[str, dict[str, int]] = {}
    for (ds, _, _), v in load_split_manifest().items():
        counts.setdefault(ds, {}).setdefault(v["split"], 0)
        counts[ds][v["split"]] += 1
    return counts
