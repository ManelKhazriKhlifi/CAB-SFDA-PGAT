"""shot.py — SHOT (Liang et al., ICML 2020)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None, verbose=True):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    optim   = make_optimizer(student, lr)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"SHOT {CFG.task} s{CFG.seed} ep {ep}/{epochs}")
        acc, obs = 0, 0
        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, _ = student(xw, return_feat=True)
                z_s, _ = student(xs, return_feat=True)
            z_w = z_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            with torch.no_grad():
                p_w = torch.softmax(z_w, dim=1)
                conf, y_hat = p_w.max(dim=1)
                mask = conf >= CFG.conf_thresh_init

            if mask.sum() > 0:
                loss_pl = F.cross_entropy(z_s[mask], y_hat[mask])
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            loss = CFG.lambda_im * loss_im + CFG.lambda_pl * loss_pl
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            obs += mask.numel(); acc += int(mask.sum().item())
            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}")

        accepts.append(acc / max(obs, 1))

    return save_results(student, "shot", accepts)