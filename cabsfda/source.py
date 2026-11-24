"""Source model training (one model per source domain x backbone x seed) and source-only evaluation.

The same source model is used by CAB-SFDA, every ablation and every baseline
for a given (task, seed), so differences between methods come from adaptation
only. The model from the last source epoch is used (no selection on any data).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_datasets, resolve_path
from .data import LabeledImages, build_transforms, make_loader, read_split
from .metrics import evaluate
from .models import build_model
from .utils import environment_info, file_lock, get_device, load_json, log, save_json, seed_everything

SOURCE_KEYS = ["arch", "pretrained", "feat_dim", "img_size", "src_epochs", "src_batch_size",
               "src_lr", "splits_dir"]


def source_dir(cfg, domain: str, seed: int) -> Path:
    return resolve_path(cfg.results_dir) / "source" / domain / cfg.arch / f"seed{seed}"


def _num_classes(cfg, domain):
    ds = load_datasets(cfg)
    return len(ds["benchmarks"][ds["domains"][domain]["benchmark"]]["classes"])


def train_source(cfg, domain: str, seed: int) -> Path:
    out = source_dir(cfg, domain, seed)
    ckpt = out / "model.pt"
    wanted = {k: cfg[k] for k in SOURCE_KEYS}
    with file_lock(out / ".lock"):
        if ckpt.exists():
            have = load_json(out / "source_config.json")
            if have != wanted:
                raise RuntimeError(f"{ckpt} was trained with {have}, current config is {wanted}. "
                                   f"Use another results_dir or delete the folder.")
            return ckpt
        seed_everything(seed, cfg.deterministic)
        device = get_device()
        K = _num_classes(cfg, domain)
        tf = build_transforms(cfg.img_size)
        tr_p, tr_y = read_split(cfg, domain, "train", role="source")
        te_p, te_y = read_split(cfg, domain, "test", role="source")
        train_loader = make_loader(LabeledImages(tr_p, tr_y, tf["source_train"]), cfg.src_batch_size,
                                   True, seed, cfg.num_workers, train=True)
        test_loader = make_loader(LabeledImages(te_p, te_y, tf["test"]), cfg.src_batch_size,
                                  False, seed, cfg.num_workers)
        model = build_model(cfg, K).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.src_lr)
        amp = cfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        history, t0 = [], time.time()
        for ep in range(1, cfg.src_epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for x, y, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = F.cross_entropy(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(y)
                n += len(y)
            history.append({"epoch": ep, "loss": tot / max(1, n)})
            if ep % 10 == 0 or ep == cfg.src_epochs:
                log(f"source {domain} {cfg.arch} seed{seed} epoch {ep}/{cfg.src_epochs} loss {tot / max(1, n):.4f}")
        metrics = evaluate(model, test_loader, K)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt.with_suffix(".tmp"))
        ckpt.with_suffix(".tmp").replace(ckpt)
        save_json(wanted, out / "source_config.json")
        save_json({"domain": domain, "seed": seed, "arch": cfg.arch, "n_train": len(tr_y),
                   "n_test": len(te_y), "source_test": metrics, "history": history,
                   "wall_time_s": time.time() - t0, "env": environment_info()}, out / "source_result.json")
        log(f"source {domain} seed{seed}: source test OA {metrics['oa']:.2f}")
        return ckpt


def load_source_model(cfg, info, seed):
    ckpt = train_source(cfg, info["source"], seed)
    c = cfg.copy()
    c.pretrained = False                      # weights come from the checkpoint
    model = build_model(c, info["num_classes"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(get_device())


def source_only_eval(cfg, info, seed, test_loader=None, save_feats=False) -> dict:
    """Source model evaluated on the target test split (cached next to the source model)."""
    out = source_dir(cfg, info["source"], seed) / f"eval_on_{info['target']}.json"
    if out.exists() and not save_feats:
        return load_json(out)
    model = load_source_model(cfg, info, seed)
    if test_loader is None:
        tf = build_transforms(cfg.img_size)
        p, y = read_split(cfg, info["target"], "test", role="target")
        test_loader = make_loader(LabeledImages(p, y, tf["test"]), cfg.batch_size, False, seed, cfg.num_workers)
    m, raw = evaluate(model, test_loader, info["num_classes"], return_raw=True, return_feats=save_feats)
    save_json(m, out)
    if save_feats:
        np.savez_compressed(out.with_name(f"features_on_{info['target']}.npz"),
                            feats=raw["feats"].astype(np.float16), labels=raw["labels"])
    return m
