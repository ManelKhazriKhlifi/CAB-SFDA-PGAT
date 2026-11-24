"""Configuration: defaults.yaml + per-run overrides, dataset/task definitions."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """Dictionary with attribute access (cfg.lr == cfg['lr'])."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return Cfg(copy.deepcopy(dict(self)))


def _expand_env(value):
    """Expand ${VAR} / ${VAR:-default} in strings (recursively)."""
    if isinstance(value, str):
        def repl(m):
            name, _, default = m.group(1).partition(":-")
            return os.environ.get(name, default if default else m.group(0))
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def parse_overrides(pairs):
    """['lr=1e-3', 'threshold=fixed'] -> {'lr': 0.001, 'threshold': 'fixed'} (YAML-typed)."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"override '{p}' must look like key=value")
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(overrides: dict | None = None, defaults_path: str | None = None) -> Cfg:
    path = Path(defaults_path or os.environ.get("CABSFDA_DEFAULTS") or REPO_ROOT / "configs" / "defaults.yaml")
    with open(path) as f:
        cfg = Cfg(yaml.safe_load(f))
    # configs/selected.yaml (written by select_hparams.py from the label-free pilot) overrides the
    # defaults for every run; CABSFDA_NO_SELECTED=1 ignores it.
    sel = REPO_ROOT / "configs" / "selected.yaml"
    if sel.exists() and not os.environ.get("CABSFDA_NO_SELECTED") and defaults_path is None:
        with open(sel) as f:
            for k, v in (yaml.safe_load(f) or {}).get("overrides", {}).items():
                if k not in cfg:
                    raise KeyError(f"{sel}: unknown config key '{k}'")
                cfg[k] = v
    # CABSFDA_SET="key=value key=value" applies to every script and every launched job
    env = parse_overrides(os.environ.get("CABSFDA_SET", "").split())
    for k, v in {**env, **(overrides or {})}.items():
        if k not in cfg:
            raise KeyError(f"unknown config key '{k}' (see configs/defaults.yaml)")
        cfg[k] = v
    return cfg


def resolve_path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


_DATASETS_CACHE = {}


def load_datasets(cfg: Cfg | None = None) -> dict:
    path = resolve_path(cfg.datasets_config if cfg else "configs/datasets.yaml")
    key = str(path)
    if key not in _DATASETS_CACHE:
        with open(path) as f:
            _DATASETS_CACHE[key] = _expand_env(yaml.safe_load(f))
    return _DATASETS_CACHE[key]


def task_info(task: str, cfg: Cfg | None = None) -> dict:
    ds = load_datasets(cfg)
    if task not in ds["tasks"]:
        raise KeyError(f"unknown task '{task}'. Known: {sorted(ds['tasks'])}")
    src, tgt, bench = ds["tasks"][task]
    classes = ds["benchmarks"][bench]["classes"]
    return {"task": task, "source": src, "target": tgt, "benchmark": bench,
            "classes": list(classes), "num_classes": len(classes)}


def all_tasks(cfg: Cfg | None = None, benchmark: str | None = None) -> list[str]:
    ds = load_datasets(cfg)
    return [t for t, (_, _, b) in ds["tasks"].items() if benchmark is None or b == benchmark]


def domain_benchmark(domain: str, cfg: Cfg | None = None) -> str:
    return load_datasets(cfg)["domains"][domain]["benchmark"]
