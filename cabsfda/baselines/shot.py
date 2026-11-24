"""SHOT (Liang et al., ICML 2020). Follows github.com/tim-learn/SHOT (object/image_target.py):
information maximisation + self-supervised pseudo-labels from prediction-weighted cosine
clustering, recomputed at the start of every epoch; CE weight cls_par = 0.3."""
import torch.nn.functional as F
import torch

from ..losses import im_loss
from .common import Ctx, EpochLog, extract, finish_epoch, pl_accuracy, shot_pseudo_labels

N_STRONG = 0


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    for ep in range(1, cfg.epochs + 1):
        feats, probs = extract(model, data.adapt_eval, ctx.device)
        pl = torch.as_tensor(shot_pseudo_labels(feats, probs), device=ctx.device)
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            x, idx = b["weak"].to(ctx.device, non_blocking=True), b["idx"].to(ctx.device)
            with ctx.autocast():
                z = model(x)
            z = z.float()
            l_cls = F.cross_entropy(z, pl[idx])
            l_im = im_loss(z)
            loss = cfg.shot_cls_par * l_cls + cfg.shot_ent_par * l_im
            ctx.step(loss)
            el.add({"total": loss.item(), "cls": l_cls.item(), "im": l_im.item()}, z.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg,
                     el.close(ep, {"pl_accuracy": pl_accuracy(pl.cpu().numpy(), data.adapt_labels)}))
    return {"history": history, "diagnostics": {}}
