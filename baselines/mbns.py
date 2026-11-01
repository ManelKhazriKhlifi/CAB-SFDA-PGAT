"""mbns.py — MBNS (Ishii & Sugiyama, 2021)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, save_results


def _bn_stats(model):
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            stats[name] = (m.running_mean.clone(), m.running_var.clone())
    return stats


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None, verbose=True):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    source_bn = _bn_stats(student)

    # Train only BN affine params
    for p in student.parameters():
        p.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    optim = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=lr)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"MBNS {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, _ = student(xw, return_feat=True)
                z_s, _ = student(xs, return_feat=True)
            z_w = z_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            loss_bn = 0.0
            for name, m in student.named_modules():
                if name in source_bn:
                    sm, sv = source_bn[name]
                    loss_bn = loss_bn + F.mse_loss(m.running_mean, sm) \
                                      + F.mse_loss(m.running_var, sv)
            loss_bn = loss_bn / max(1, len(source_bn))

            with torch.no_grad():
                p_w = torch.softmax(z_w / CFG.temp_cons, dim=1)
            loss_pl = -(p_w * F.log_softmax(z_s, dim=1)).sum(dim=1).mean()

            loss = CFG.lambda_im * loss_im + loss_bn + CFG.lambda_pl * loss_pl
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 BN=f"{loss_bn.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}")

        accepts.append(1.0)

    return save_results(student, "mbns", accepts)