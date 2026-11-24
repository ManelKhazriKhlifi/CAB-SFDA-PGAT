"""SMNA (Zhu et al., IEEE TGRS 2025): source-free cross-domain scene classification via statistics
matching and noise adaptation.

No official code was available to us; this re-implementation follows the paper. The paper describes a
model-statistics-guided alignment (target feature statistics aligned to the statistics stored in
the source model) and a noise-adaptation module that limits the effect of noisy pseudo-labels.
This re-implementation uses (i) MSE between the target batch statistics at every BatchNorm
input and the stored source statistics, (ii) label-smoothed cross-entropy on confident
(temperature-sharpened) pseudo-labels on the strong view, and (iii) information maximisation.
It may differ from the authors' implementation."""
import torch.nn.functional as F

from ..losses import im_loss
from .common import BNStats, Ctx, EpochLog, finish_epoch

N_STRONG = 1


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    bn = BNStats(model)
    for ep in range(1, cfg.epochs + 1):
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            xw, xs = b["weak"].to(ctx.device, non_blocking=True), b["strong"].to(ctx.device, non_blocking=True)
            bn.enabled = True
            with ctx.autocast():
                zw = model(xw)
            bn.enabled = False
            with ctx.autocast():
                zs = model(xs)
            zw, zs = zw.float(), zs.float()
            p = (zw.detach() / cfg.smna_temp).softmax(1)
            conf, y = p.max(1)
            keep = conf >= cfg.smna_tau
            l_pl = (F.cross_entropy(zs[keep], y[keep], label_smoothing=cfg.smna_label_smoothing)
                    if keep.any() else zs.new_zeros(()))
            l_bn = bn.mse()
            l_im = im_loss(zw)
            loss = l_im + cfg.smna_lambda_bn * l_bn + cfg.smna_lambda_pl * l_pl
            ctx.step(loss)
            el.add({"total": loss.item(), "im": l_im.item(), "bn": l_bn.item(), "pl": l_pl.item(),
                    "accept_rate": 100.0 * float(keep.float().mean())}, zw.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg, el.close(ep))
    bn.remove()
    return {"history": history, "diagnostics": {}}
