"""Single entry point for reading `config/runtime*.json` and resolving all
paths from environment variables.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_RUNTIME_CONFIG = PACKAGE_ROOT / "config" / "runtime.json"

def runtime_config_path() -> Path:
    return Path(os.environ.get("SELFMORPH_RUNTIME_CONFIG", str(_DEFAULT_RUNTIME_CONFIG)))

@lru_cache(maxsize=1)
def load_runtime_config() -> dict[str, Any]:
    path = runtime_config_path()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))

def dataset_root() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["dataset_root_env"]
    return _env_path(env_key, str(PACKAGE_ROOT / "resources" / "datasets"))

def eval_dataset_root() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["eval_dataset_root_env"]
    return _env_path(env_key, str(PACKAGE_ROOT / "resources" / "eval_datasets"))

def artifacts_dir() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["artifacts_dir_env"]
    return _env_path(env_key, str(PACKAGE_ROOT / "artifacts"))

def oracle_dir() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["oracle_dir_env"]
    return _env_path(env_key, str(artifacts_dir() / "oracle"))

def training_data_dir() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["training_data_dir_env"]
    return _env_path(env_key, str(artifacts_dir() / "training_data"))

def checkpoint_dir() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["checkpoint_dir_env"]
    return _env_path(env_key, str(artifacts_dir() / "checkpoints"))

def results_dir() -> Path:
    cfg = load_runtime_config()
    env_key = cfg["paths"]["results_dir_env"]
    return _env_path(env_key, str(artifacts_dir() / "results"))

def model_path() -> str:
    cfg = load_runtime_config()
    env_key = cfg["model"]["model_path_env"]
    value = os.environ.get(env_key)
    if not value:
        raise RuntimeError(
            f"{env_key} is not set. Copy .env.example to .env and fill in the "
            f"frozen base LM path."
        )
    return value
