"""Shared optimisation protocol for CAB-SFDA and every baseline.

Default: the recipe of the official SHOT / NRC / AaD / G-SFDA code - SGD with momentum 0.9,
Nesterov, weight decay 1e-3, lr 1e-2 on the projection head and lr x 0.1 on the pretrained
backbone, decayed per iteration as lr * (1 + 10 p)^-0.75 (p = iteration / total iterations),
plus gradient-norm clipping.

`optimizer: adam` is also available (with `backbone_lr_mult`, `lr_schedule: none` and `grad_clip: 0`
to disable the other parts of the protocol).
"""
from __future__ import annotations

import torch


def param_groups(model, cfg):
    """backbone (model.features) at lr * backbone_lr_mult; projection head (and the classifier
    when it is trainable) at lr. Frozen parameters are excluded."""
    for p in model.classifier.parameters():
        p.requires_grad_(not cfg.freeze_classifier)
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    head = [p for m in (model.feature_head, model.classifier) for p in m.parameters() if p.requires_grad]
    groups = [{"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult, "name": "backbone"},
              {"params": head, "lr": cfg.lr, "name": "head"}]
    for g in groups:
        g["initial_lr"] = g["lr"]
    return [g for g in groups if g["params"]]


def build_optimizer(model, cfg):
    groups = param_groups(model, cfg)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.momentum > 0,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"optimizer must be 'sgd' or 'adam', got {cfg.optimizer}")


class Stepper:
    """Optimizer + AMP scaler + per-iteration lr schedule + gradient clipping."""

    def __init__(self, model, cfg, iters_per_epoch: int, device):
        self.model, self.cfg = model, cfg
        self.opt = build_optimizer(model, cfg)
        self.amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.total = max(1, iters_per_epoch * cfg.epochs)
        self.it = 0
        self.skipped = 0            # steps skipped by the AMP scaler (inf/nan gradients)

    def lr_factor(self):
        if self.cfg.lr_schedule == "poly":
            return (1 + 10 * self.it / self.total) ** (-0.75)
        if self.cfg.lr_schedule == "none":
            return 1.0
        raise ValueError(f"lr_schedule {self.cfg.lr_schedule}")

    def step(self, loss):
        for g in self.opt.param_groups:
            g["lr"] = g["initial_lr"] * self.lr_factor()
        self.opt.zero_grad(set_to_none=True)
        if loss.requires_grad and torch.isfinite(loss):
            self.scaler.scale(loss).backward()
            finite = True
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                params = [p for g in self.opt.param_groups for p in g["params"]]
                finite = bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)))
            if self.amp:                      # the scaler skips non-finite steps itself
                self.scaler.step(self.opt)
                self.scaler.update()
                self.skipped += int(not finite)
            elif finite:
                self.opt.step()
            else:
                self.skipped += 1
        elif loss.requires_grad:
            self.skipped += 1                 # non-finite loss: no update
        self.it += 1

    def current_lr(self):
        return {g.get("name", str(i)): g["lr"] for i, g in enumerate(self.opt.param_groups)}


@torch.no_grad()
def ema_update(teacher, student, momentum: float):
    """teacher <- m * teacher + (1 - m) * student (parameters); buffers (BN statistics) copied."""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)
