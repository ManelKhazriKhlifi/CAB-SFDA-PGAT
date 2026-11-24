"""NRC (Yang et al., NeurIPS 2021). Follows github.com/Albert0147/NRC_SFDA (train_tar.py):
feature/score memory banks over all adaptation images, K nearest neighbours weighted 1 if
reciprocal (0.1 otherwise), KK expanded neighbours weighted 0.1, plus the diversity term."""
import torch
import torch.nn.functional as F

from .common import Ctx, EpochLog, extract, finish_epoch

N_STRONG = 0


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    feats, probs = extract(model, data.adapt_eval, ctx.device)
    fea_bank, score_bank = F.normalize(feats, dim=1), probs.clone()
    K, KK = cfg.nrc_k, cfg.nrc_kk
    for ep in range(1, cfg.epochs + 1):
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            x, idx = b["weak"].to(ctx.device, non_blocking=True), b["idx"].to(ctx.device)
            with ctx.autocast():
                z, f = model(x, return_feat=True)
            z, f = z.float(), f.float()
            p = z.softmax(1)
            with torch.no_grad():
                fn = F.normalize(f, dim=1)
                fea_bank[idx], score_bank[idx] = fn, p.detach()
                _, near = (fn @ fea_bank.T).topk(K + 1, dim=1)
                near = near[:, 1:]                                          # B x K
                score_near = score_bank[near]                               # B x K x C
                _, nn2 = torch.bmm(fea_bank[near], fea_bank.T.unsqueeze(0).expand(len(x), -1, -1)
                                   ).topk(KK + 1, dim=-1)
                nn2 = nn2[:, :, 1:]                                         # B x K x KK
                reciprocal = (nn2 == idx[:, None, None]).any(-1)
                w = torch.where(reciprocal, torch.ones_like(reciprocal, dtype=torch.float),
                                torch.full_like(reciprocal, 0.1, dtype=torch.float))
                score_kk = score_bank[nn2].reshape(len(x), K * KK, -1)      # B x K*KK x C
            l_exp = -(0.1 * (p[:, None, :] * score_kk).sum(-1)).sum(1).mean()
            l_nei = -(w * (p[:, None, :] * score_near).sum(-1)).sum(1).mean()
            m = p.mean(0)
            l_div = (m * torch.log(m + 1e-5)).sum()
            loss = l_exp + l_nei + l_div
            ctx.step(loss)
            el.add({"total": loss.item(), "neighbors": l_nei.item(), "expanded": l_exp.item(),
                    "div": l_div.item()}, z.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg, el.close(ep))
    return {"history": history, "diagnostics": {}}
