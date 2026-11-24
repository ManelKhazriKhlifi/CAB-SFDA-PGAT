"""G-SFDA (Yang et al., ICCV 2021), local structure clustering (LSC) part. Follows
github.com/Albert0147/G-SFDA (train_tar_oh.py): neighbours retrieved from memory banks that
exclude the current mini-batch, loss -log(p . s_k) over the k=2 nearest neighbours plus the
diversity term. Sparse domain attention (SDA) only serves to keep source accuracy and needs a
source model trained with attention masks; it is not used here (it does not target target
accuracy)."""
import torch
import torch.nn.functional as F

from .common import Ctx, EpochLog, extract, finish_epoch

N_STRONG = 0


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    feats, probs = extract(model, data.adapt_eval, ctx.device)
    fea_bank, score_bank = F.normalize(feats, dim=1), probs.clone()
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
                fea_bank[idx] = -0.1                 # exclude the current mini-batch
                fn = F.normalize(f, dim=1)
                _, near = (fn @ fea_bank.T).topk(cfg.gsfda_k, dim=1)
                score_near = score_bank[near]        # B x k x C
                fea_bank[idx], score_bank[idx] = fn, p.detach()
            l_const = -torch.log((p[:, None, :] * score_near).sum(-1) + 1e-8).sum(1).mean()
            m = p.mean(0)
            l_div = (m * torch.log(m + 1e-5)).sum()
            loss = l_const + l_div
            ctx.step(loss)
            el.add({"total": loss.item(), "lsc": l_const.item(), "div": l_div.item()}, z.detach(), b["y_hidden"])
        finish_epoch(logger, history, info, cfg, el.close(ep))
    return {"history": history, "diagnostics": {}}
