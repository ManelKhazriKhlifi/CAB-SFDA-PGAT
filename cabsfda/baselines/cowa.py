"""CoWA-JMDS (Lee et al., ICML 2022). Follows github.com/Jhyun17/CoWA-JMDS (image_target_CoWA.py):
at the start of training and of every epoch, fit a GMM (per-class covariance) on L2-normalised
features initialised from the model's predictions, take its posteriors as pseudo-labels and
the JMDS score (log-probability gap x model probability of the pseudo-label) as sample weight;
train with weight mixup (Beta(alpha, alpha))."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .common import Ctx, EpochLog, extract, finish_epoch, pl_accuracy

N_STRONG = 0


def _gmm(x, pi, mu, resp, eps):
    log_probs = []
    d = x.shape[1]
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    for i in range(len(mu)):
        t = x - mu[i]
        w = resp[:, i:i + 1]
        cov = (t.T @ (t * w)) / w.sum().clamp_min(1e-8) + eps * eye
        try:
            chol = torch.linalg.cholesky(cov)
        except RuntimeError:
            cov = cov + 100 * eps * eye
            chol = torch.linalg.cholesky(cov)
        logdet = 2 * torch.log(torch.diagonal(chol)).sum()
        sol = torch.cholesky_solve(t.T, chol).T
        mah = (t * sol).sum(1)
        log_probs.append(-0.5 * (d * math.log(2 * math.pi) + logdet + mah) + torch.log(pi[i].clamp_min(1e-8)))
    log_probs = torch.stack(log_probs, 1)
    zz = log_probs - torch.logsumexp(log_probs, 1, keepdim=True)
    return zz, zz.exp()


@torch.no_grad()
def jmds_labels(model, data, cfg, device):
    feats, probs = extract(model, data.adapt_eval, device)
    x = F.normalize(feats.double(), dim=1)
    probs = probs.double()
    K = probs.shape[1]
    uniform = torch.full_like(probs, 1.0 / K)
    pi = probs.sum(0)
    mu = (probs.T @ x) / pi[:, None].clamp_min(1e-8)
    zz, gamma = _gmm(x, pi, mu, uniform, cfg.cowa_eps)
    pi = gamma.sum(0)
    mu = (gamma.T @ x) / pi[:, None].clamp_min(1e-8)
    zz, gamma = _gmm(x, pi, mu, gamma, cfg.cowa_eps)
    pred = gamma.argmax(1)
    top2 = zz.sort(1, descending=True)[0]
    gap = top2[:, 0] - top2[:, 1]
    lpg = gap / gap.max().clamp_min(1e-12)
    ppl = probs.gather(1, pred[:, None]).squeeze(1)
    coeff = {"JMDS": lpg * ppl, "LPG": lpg, "PPL": ppl}[cfg.cowa_coeff]
    return pred.to(device), coeff.float().to(device)


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    K = info["num_classes"]
    rng = np.random.default_rng(int(torch.initial_seed()) % 2 ** 32)
    for ep in range(1, cfg.epochs + 1):
        pl, coeff = jmds_labels(model, data, cfg, ctx.device)
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            x, idx = b["weak"].to(ctx.device, non_blocking=True), b["idx"].to(ctx.device)
            t = F.one_hot(pl[idx], K).float()
            c = coeff[idx]
            lam = torch.as_tensor(rng.beta(cfg.cowa_alpha, cfg.cowa_alpha, len(x)), dtype=torch.float,
                                  device=ctx.device)
            perm = torch.randperm(len(x), device=ctx.device)
            xm = lam[:, None, None, None] * x + (1 - lam[:, None, None, None]) * x[perm]
            tm = lam[:, None] * t + (1 - lam[:, None]) * t[perm]
            cm = lam * c + (1 - lam) * c[perm]
            with ctx.autocast():
                z = model(xm)
            z = z.float()
            loss = ((-tm * torch.log(z.softmax(1) + 1e-6)).sum(1) * cm).mean()
            ctx.step(loss)
            el.add({"total": loss.item()}, None, None)
        finish_epoch(logger, history, info, cfg,
                     el.close(ep, {"pl_accuracy": pl_accuracy(pl.cpu().numpy(), data.adapt_labels),
                                   "mean_coeff": float(coeff.mean())}))
    return {"history": history, "diagnostics": {}}
