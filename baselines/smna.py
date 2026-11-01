"""smna.py — SMNA (Zhu et al., IEEE TGRS 2025)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, smoothing=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)

    src_stats = {}
    for name, m in student.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            src_stats[name] = (m.running_mean.clone(), m.running_var.clone())

    # BN alignment needs the CURRENT batch statistics (with gradient).
    # running_mean/var are buffers without grad, so capture them via hooks.
    bn_batch = {}

    def _bn_hook(name):
        def fn(mod, inp, out):
            x = inp[0].float()
            dims = [0] if x.dim() == 2 else [0, 2, 3]
            bn_batch[name] = (x.mean(dim=dims), x.var(dim=dims, unbiased=False))
        return fn

    hooks = [m.register_forward_hook(_bn_hook(n))
             for n, m in student.named_modules() if n in src_stats]

    optim = torch.optim.Adam(
        [p for p in list(student.features.parameters())
                    + list(student.feature_head.parameters())], lr=lr)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"SMNA {CFG.task} s{CFG.seed} ep {ep}/{epochs}")
        acc, obs = 0, 0

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                bn_batch.clear()
                z_w, _ = student(xw, return_feat=True)
                stats_w = dict(bn_batch)          # weak-view batch statistics
                z_s, _ = student(xs, return_feat=True)
            z_w = z_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            with torch.no_grad():
                p_w = torch.softmax(z_w / CFG.temp_cons, dim=1)
                conf, y_hat = p_w.max(dim=1)
                mask = conf >= CFG.conf_thresh_init

            if mask.sum() > 0:
                n = int(mask.sum())
                K = CFG.num_classes
                smooth = torch.full((n, K), smoothing / K, device=DEVICE)
                smooth.scatter_(1, y_hat[mask].unsqueeze(1),
                                1.0 - smoothing + smoothing / K)
                loss_pl = -(smooth * F.log_softmax(z_s[mask], dim=1)) \
                              .sum(dim=1).mean()
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            loss_bn = torch.tensor(0.0, device=DEVICE)
            for name, (bm, bv) in stats_w.items():
                sm, sv = src_stats[name]
                loss_bn = loss_bn + F.mse_loss(bm, sm) + F.mse_loss(bv, sv)
            loss_bn = loss_bn / max(1, len(stats_w))

            loss = (CFG.lambda_im * loss_im
                    + CFG.lambda_pl * loss_pl
                    + 0.1 * loss_bn)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            obs += mask.numel(); acc += int(mask.sum().item())
            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}",
                                 BN=f"{loss_bn.item():.3f}")

        accepts.append(acc / max(obs, 1))

    for h in hooks:
        h.remove()
    return save_results(student, "smna", accepts)