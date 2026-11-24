"""AaD (Yang et al., NeurIPS 2022). Follows github.com/Albert0147/AaD_SFDA:
attract the prediction to those of its K nearest neighbours (memory banks) and disperse it
from the other predictions in the batch; alpha decay as in the official code (off by default,
office-home setting)."""
import torch
import torch.nn.functional as F

from .common import Ctx, EpochLog, extract, finish_epoch

N_STRONG = 0


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    feats, probs = extract(model, data.adapt_eval, ctx.device)
    fea_bank, score_bank = F.normalize(feats, dim=1), probs.clone()
    max_iter, it = cfg.epochs * len(data.train), 0
    for ep in range(1, cfg.epochs + 1):
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            x, idx = b["weak"].to(ctx.device, non_blocking=True), b["idx"].to(ctx.device)
            it += 1
            alpha = ((1 + 10 * it / max_iter) ** (-cfg.aad_beta) * cfg.aad_alpha
                     if cfg.aad_alpha_decay else cfg.aad_alpha)
            with ctx.autocast():
                z, f = model(x, return_feat=True)
            z, f = z.float(), f.float()
            p = z.softmax(1)
            with torch.no_grad():
                fn = F.normalize(f, dim=1)
                fea_bank[idx], score_bank[idx] = fn, p.detach()
                _, near = (fn @ fea_bank.T).topk(cfg.aad_k + 1, dim=1)
                score_near = score_bank[near[:, 1:]]
            l_att = -(p[:, None, :] * score_near).sum(-1).sum(1).mean()
            off_diag = 1 - torch.eye(len(x), device=ctx.device)
            l_dis = ((p @ p.T) * off_diag).sum(-1).mean()
            loss = l_att + alpha * l_dis
            ctx.step(loss)
            el.add({"total": loss.item(), "attract": l_att.item(), "disperse": l_dis.item()},
                   z.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg, el.close(ep))
    return {"history": history, "diagnostics": {}}
