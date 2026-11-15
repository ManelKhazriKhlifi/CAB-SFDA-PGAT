import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from smna import _gauss_kl
from _common import load_student, save_results, RunLogger


def _bn_stats(model):
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            stats[name] = (m.running_mean.clone(), m.running_var.clone())
    return stats


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None, verbose=True,
        lambda_bn=1.0, lambda_pl=CFG.lambda_pl, backbone_lr_mult=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    source_bn = _bn_stats(student)

    bn_batch = {}

    def _bn_hook(name):
        def fn(mod, inp, out):
            x = inp[0].float()
            dims = [0] if x.dim() == 2 else [0, 2, 3]
            bn_batch[name] = (x.mean(dim=dims), x.var(dim=dims, unbiased=False))
        return fn

    hooks = [m.register_forward_hook(_bn_hook(n))
             for n, m in student.named_modules() if n in source_bn]

    optim = torch.optim.Adam([
        {"params": student.features.parameters(), "lr": lr * backbone_lr_mult},
        {"params": student.feature_head.parameters(), "lr": lr}])
    logger = RunLogger("mbns", src_ckpt_path,
                       {"arch": arch, "epochs": epochs, "lr": lr,
                        "lambda_bn": lambda_bn, "lambda_pl": lambda_pl,
                        "backbone_lr_mult": backbone_lr_mult})

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
            bn_batch.clear()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, _ = student(xw, return_feat=True)
                stats_w = dict(bn_batch)          # weak-view batch statistics
                z_s, _ = student(xs, return_feat=True)
            z_w = z_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            loss_bn = torch.tensor(0.0, device=DEVICE)
            for name, (bm, bv) in stats_w.items():
                sm, sv = source_bn[name]
                loss_bn = loss_bn + _gauss_kl(bm, bv, sm, sv)
            loss_bn = loss_bn / max(1, len(stats_w))

            # Soft pseudo-label: sharpened weak-view prediction -> strong view
            with torch.no_grad():
                p_w = torch.softmax(z_w / CFG.temp_cons, dim=1)
            loss_pl = -(p_w * F.log_softmax(z_s, dim=1)).sum(dim=1).mean()

            loss = (CFG.lambda_im * loss_im + lambda_bn * loss_bn
                    + lambda_pl * loss_pl)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            logger.step(ep, im=loss_im, bn=loss_bn, pl=loss_pl, total=loss)
            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 BN=f"{loss_bn.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}")

        accepts.append(1.0)
        logger.end_epoch(ep, accepts[-1], student)

    for h in hooks:
        h.remove()
    return save_results(student, "mbns", accepts, logger=logger)
