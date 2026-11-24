"""Seeding, device, JSON I/O, file locks, logging helpers."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, cls=_NumpyEncoder)
    os.replace(tmp, path)          # atomic: a result file is either complete or absent


def load_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, cls=_NumpyEncoder) + "\n")


@contextlib.contextmanager
def file_lock(path):
    """Exclusive inter-process lock (used so parallel jobs never train the same source model twice)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def environment_info() -> dict:
    import torchvision
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def set_bn_mode(model: torch.nn.Module, mode: str):
    """Call after model.train(). 'frozen' keeps BN running statistics fixed (eval mode)."""
    if mode == "frozen":
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    elif mode != "train":
        raise ValueError(f"bn_mode must be 'frozen' or 'train', got {mode}")
