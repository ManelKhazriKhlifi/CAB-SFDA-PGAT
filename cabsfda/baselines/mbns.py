"""MBNS (Ishii & Sugiyama, 2021): source-free domain adaptation by matching the batch-normalisation
statistics stored in the source model.

No official code is available; implemented from the paper: the target batch statistics at the
input of every BatchNorm layer are aligned to the stored source statistics with a Gaussian
KL divergence (computed with gradient through the feature extractor), together with
information maximisation. The source statistics are copied before adaptation, so they stay
fixed even when bn_mode=train.

The matching term is computed from the statistics of the current batch, with gradient (BN running
buffers carry no gradient and cannot be used for this term)."""
from ..losses import im_loss
from .common import BNStats, Ctx, EpochLog, finish_epoch

N_STRONG = 0


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    bn = BNStats(model)
    for ep in range(1, cfg.epochs + 1):
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            x = b["weak"].to(ctx.device, non_blocking=True)
            bn.enabled = True
            with ctx.autocast():
                z = model(x)
            bn.enabled = False
            z = z.float()
            l_bn = bn.gaussian_kl()
            l_im = im_loss(z)
            loss = l_im + cfg.mbns_lambda_bn * l_bn
            ctx.step(loss)
            el.add({"total": loss.item(), "im": l_im.item(), "bn_kl": l_bn.item()}, z.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg, el.close(ep))
    bn.remove()
    return {"history": history, "diagnostics": {}}
