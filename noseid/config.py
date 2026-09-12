"""Configuration loading.

A single ``get_config()`` returns a dict-from-YAML merged over the defaults.
Deeply nested updates are supported so callers can pass a small override dict.
"""
from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

_cfg_cache: dict | None = None


def _deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(overrides: Mapping[str, Any] | None = None, *,
               reload: bool = False) -> dict:
    """Return the active configuration dict.

    ``overrides`` is deep-merged on top of the cached defaults; pass
    ``reload=True`` to re-read the YAML from disk (discards the cache).
    """
    global _cfg_cache
    if _cfg_cache is None or reload:
        _cfg_cache = load_yaml(CONFIG_PATH)
    cfg = _cfg_cache if overrides is None else _deep_merge(_cfg_cache, overrides)
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a dotted path, e.g. cfg_get(cfg, 'embedding.dim')."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
