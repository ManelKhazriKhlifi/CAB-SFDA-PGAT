"""DaC - Divide and Contrast (Zhang et al., NeurIPS 2022).

Independent implementation of the algorithm described in the paper and released at
github.com/ZyeZhang/DaC (GPL; no code copied):
  * SHOT pseudo-labels (recomputed every epoch), CE on the weak and two strong views (x 0.65),
    plus information maximisation;
  * after a one-epoch warm-up, a memory bank of L2-normalised features; the top-k most
    confident samples of each class (and later any sample with confidence >= p_threshold)
    are 'source-like' and carry their class label, the others are 'target-specific' and
    carry their own instance label;
  * divide-and-contrast loss: softmax over groups (class groups for source-like samples,
    instance groups for target-specific samples, each target-specific group extended with its
    two strong views and k nearest memory neighbours);
  * alignment loss pulling each sample towards the other group (source-like <-> target-specific)
    of its pseudo-class;
  * momentum update of the memory bank.
"""
import torch
import torch.nn.functional as F

from ..losses import im_loss
from .common import Ctx, EpochLog, extract, finish_epoch, pl_accuracy, shot_pseudo_labels

N_STRONG = 2


def _masked_softmax(v, mask, eps=1e-6):
    e = torch.exp(v - v.max(1, keepdim=True)[0]) * mask
    return e / (e.sum(1, keepdim=True) + eps)


def adapt(model, data, cfg, info, logger):
    ctx, history = Ctx(model, cfg, len(data.train)), []
    C, N, dev = info["num_classes"], data.n_adapt, ctx.device
    bank = labels = None
    topk = max(1, round(cfg.dac_topk_frac * N / C))
    for ep in range(1, cfg.epochs + 1):
        feats, probs = extract(model, data.adapt_eval, dev)
        pl = torch.as_tensor(shot_pseudo_labels(feats, probs), device=dev)
        if ep == 2:                                    # memory initialised after one warm-up epoch
            bank = F.normalize(feats, dim=1)
            labels = C + torch.arange(N, device=dev)
            for c in range(C):
                labels[probs[:, c].topk(min(topk, N)).indices] = c
        ctx.train_mode()
        el = EpochLog()
        for b in data.train:
            idx = b["idx"].to(dev)
            xw, xs1, xs2 = (b[k].to(dev, non_blocking=True) for k in ("weak", "strong", "strong1"))
            with ctx.autocast():
                zw, fw = model(xw, return_feat=True)
                zs1, fs1 = model(xs1, return_feat=True)
                zs2, fs2 = model(xs2, return_feat=True)
            zw, zs1, zs2 = zw.float(), zs1.float(), zs2.float()
            y = pl[idx]
            l_cls = 0.65 * cfg.dac_cls_par * (F.cross_entropy(zw, y) + F.cross_entropy(zs1, y)
                                              + F.cross_entropy(zs2, y))
            l_im = im_loss(zw)
            loss = l_cls + l_im
            l_con = l_ad = torch.zeros((), device=dev)
            if bank is not None:
                with torch.no_grad():                  # confident samples become source-like
                    conf, cls = zw.softmax(1).max(1)
                    sure = conf >= cfg.dac_p_threshold
                    labels[idx[sure]] = cls[sure]
                fw_n, fs1_n, fs2_n = (F.normalize(t.float(), dim=1) for t in (fw, fs1, fs2))
                sim = fw_n @ bank.T / cfg.dac_temp                              # B x N
                B = len(idx)
                own = labels[idx]
                L = int(labels.max()) + 1
                # group-level similarities: sum over members, then mean
                gsum = torch.zeros(L, B, device=dev).index_add_(0, labels, sim.T)
                nums = torch.zeros(L, device=dev).index_add_(0, labels, torch.ones(N, device=dev))
                with torch.no_grad():
                    nn_idx = sim.topk(cfg.dac_k_nbor, dim=1).indices
                local = (((fw_n * fs1_n).sum(1) + (fw_n * fs2_n).sum(1)) / 2
                         + (fw_n[:, None, :] * bank[nn_idx]).sum(-1).sum(1)) / cfg.dac_temp
                gsum = gsum.index_put((own, torch.arange(B, device=dev)), local, accumulate=True)
                present = torch.zeros(L, device=dev).index_add_(0, own, torch.ones(B, device=dev)) > 0
                nums = nums + present.float() * (cfg.dac_k_nbor + 1)
                gmean = (gsum / nums.clamp_min(1)[:, None]).T                    # B x L
                p = _masked_softmax(gmean, (nums > 0).float()[None, :])
                l_con = F.nll_loss(torch.log(p + 1e-6), own)
                # alignment between the two groups of the same pseudo-class
                src_like = labels < C
                ps = pl                                                          # pseudo-class of members
                def class_mean(member_mask):
                    s = torch.zeros(C, B, device=dev).index_add_(0, ps[member_mask], sim[:, member_mask].T)
                    n = torch.zeros(C, device=dev).index_add_(0, ps[member_mask],
                                                              torch.ones(int(member_mask.sum()), device=dev))
                    return (s / n.clamp_min(1)[:, None]).T                       # B x C
                ad_s = class_mean(src_like).gather(1, y[:, None])
                ad_t = class_mean(~src_like).gather(1, y[:, None])
                target_side = (own < C).long()     # source-like -> towards target-specific, and vice versa
                l_ad = F.cross_entropy(torch.cat([ad_s, ad_t], 1), target_side)
                loss = loss + cfg.dac_lambda_m * l_con + cfg.dac_lambda_ad * l_ad
            ctx.step(loss)
            if bank is not None:
                with torch.no_grad():
                    upd = cfg.dac_momentum * bank[idx] + (1 - cfg.dac_momentum) * F.normalize(fw.float(), dim=1)
                    bank[idx] = F.normalize(upd, dim=1)
            el.add({"total": loss.item(), "cls": l_cls.item(), "im": l_im.item(), "contrast": l_con.item(),
                    "align": l_ad.item()}, zw.detach(), b["y_hidden"])
        extra = {"pl_accuracy": pl_accuracy(pl.cpu().numpy(), data.adapt_labels)}
        if labels is not None:
            extra["source_like_fraction"] = float((labels < C).float().mean())
        finish_epoch(logger, history, info, cfg, el.close(ep, extra))
    return {"history": history, "diagnostics": {}}
