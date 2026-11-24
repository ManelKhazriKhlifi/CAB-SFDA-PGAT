"""Shared protocol for all baselines.

Every baseline uses the same source model, splits, epochs, batch size, optimiser and
learning-rate schedule (cabsfda/optim.py: SGD, head lr 1e-2, backbone x 0.1, poly decay,
gradient clipping), frozen classifier and BatchNorm policy (cfg.bn_mode) as CAB-SFDA.
Only the adaptation objective differs. Each file names the official repository its algorithm follows.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from ..optim import Stepper
from ..utils import get_device, log, set_bn_mode


class Ctx:
    """Optimizer, AMP and train-mode handling shared by the baseline loops."""

    def __init__(self, model, cfg, iters_per_epoch):
        self.model, self.cfg = model, cfg
        self.device = get_device()
        self.stepper = Stepper(model, cfg, iters_per_epoch, self.device)
        self.opt = self.stepper.opt
        self.amp = self.stepper.amp

    def train_mode(self):
        self.model.train()
        set_bn_mode(self.model, self.cfg.bn_mode)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.amp)

    def step(self, loss):
        self.stepper.step(loss)


@torch.no_grad()
def extract(model, loader, device):
    """Eval-mode pass over the adaptation images: raw features and softmax outputs."""
    model.eval()
    feats, probs = [], []
    for x, _ in loader:
        z, f = model(x.to(device, non_blocking=True), return_feat=True)
        feats.append(f.float())
        probs.append(z.float().softmax(1))
    return torch.cat(feats), torch.cat(probs)


def shot_pseudo_labels(feats: torch.Tensor, probs: torch.Tensor, threshold: int = 0) -> np.ndarray:
    """Self-supervised pseudo-labels of SHOT (Liang et al., 2020; tim-learn/SHOT obtain_label):
    cosine k-means initialised with prediction-weighted centroids, then one hard-assignment round."""
    f = torch.cat([feats.cpu(), torch.ones(len(feats), 1)], 1)
    f = (f / f.norm(dim=1, keepdim=True)).numpy()
    aff = probs.cpu().numpy()
    K = aff.shape[1]
    predict = aff.argmax(1)
    for _ in range(2):
        initc = aff.T @ f / (1e-8 + aff.sum(0)[:, None])
        cls_count = np.eye(K)[predict].sum(0)
        labelset = np.where(cls_count > threshold)[0]
        dd = cdist(f, initc[labelset], "cosine")
        predict = labelset[dd.argmin(1)]
        aff = np.eye(K)[predict]
    return predict


class EpochLog:
    """Per-epoch loss averages; accuracy of argmax / pseudo-labels against hidden labels (logging only)."""

    def __init__(self):
        self.sums, self.n, self.correct, self.seen, self.t0 = {}, 0, 0, 0, time.time()

    def add(self, losses: dict, logits=None, y_hidden=None):
        for k, v in losses.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1
        if logits is not None and y_hidden is not None:
            self.correct += int((logits.argmax(1).cpu() == y_hidden).sum())
            self.seen += len(y_hidden)

    def close(self, ep, extra=None):
        out = {"epoch": ep, "losses": {k: v / max(1, self.n) for k, v in self.sums.items()},
               "argmax_accuracy_on_adaptation_batches": 100.0 * self.correct / max(1, self.seen),
               "seconds": time.time() - self.t0}
        out.update(extra or {})
        return out


def pl_accuracy(pl, labels_hidden):
    """Accuracy of offline pseudo-labels (hidden labels; logging only)."""
    return 100.0 * float((np.asarray(pl) == np.asarray(labels_hidden)).mean())


def finish_epoch(logger, history, info, cfg, rec):
    history.append(rec)
    logger(rec)
    loss = rec["losses"].get("total", float("nan"))
    log(f"{info['task']} ep {rec['epoch']}/{cfg.epochs} loss {loss:.4f}"
        + (f" PL-acc {rec['pl_accuracy']:.1f}%" if rec.get("pl_accuracy") is not None else ""))


class BNStats:
    """Captures the batch statistics of every BatchNorm input during the forward pass (with gradient)
    and keeps a copy of the source running statistics (the stored source statistics)."""

    def __init__(self, model):
        self.layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.src = [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in self.layers]
        self.cur = [None] * len(self.layers)
        self.handles = [m.register_forward_pre_hook(self._hook(i)) for i, m in enumerate(self.layers)]
        self.enabled = False

    def _hook(self, i):
        def fn(module, inp):
            if not self.enabled:
                return
            x = inp[0].float()
            dims = [0, 2, 3] if x.dim() == 4 else [0]
            self.cur[i] = (x.mean(dims), x.var(dims, unbiased=False))
        return fn

    def gaussian_kl(self):
        """mean over layers/channels of KL(N(mu_t, s_t^2) || N(mu_s, s_s^2))."""
        terms = []
        for (ms, vs), cur in zip(self.src, self.cur):
            if cur is None:
                continue
            mt, vt = cur
            vs = vs + 1e-5
            vt = vt + 1e-5
            terms.append((0.5 * (torch.log(vs / vt) + (vt + (mt - ms) ** 2) / vs - 1)).mean())
        return torch.stack(terms).mean()

    def mse(self):
        terms = []
        for (ms, vs), cur in zip(self.src, self.cur):
            if cur is None:
                continue
            mt, vt = cur
            terms.append(F.mse_loss(mt, ms) + F.mse_loss(vt, vs))
        return torch.stack(terms).mean()

    def remove(self):
        for h in self.handles:
            h.remove()
