"""aad.py — AaD (Yang et al., NeurIPS 2022)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import kl_with_temperature
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, lambda_attract=1.0, lambda_disperse=1.0,
        lambda_cons=1.0):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    optim   = make_optimizer(student, lr)

    centroids   = torch.zeros(CFG.num_classes, CFG.feat_dim, device=DEVICE)
    counts      = torch.zeros(CFG.num_classes, device=DEVICE)
    initialized = torch.zeros(CFG.num_classes, dtype=torch.bool, device=DEVICE)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"AaD {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, f_w = student(xw, return_feat=True)
                z_s, f_s = student(xs, return_feat=True)
            z_w = z_w.float(); f_w = f_w.float()
            z_s = z_s.float(); f_s = f_s.float()

            with torch.no_grad():
                y_hat = z_w.argmax(dim=1)

            # Attraction to own centroid
            with torch.no_grad():
                c = centroids[y_hat]
                valid = initialized[y_hat]
            if valid.any():
                loss_attract = F.mse_loss(f_w[valid], c[valid])
            else:
                loss_attract = torch.tensor(0.0, device=DEVICE)

            # Dispersion from other centroids
            f_norm = F.normalize(f_s, dim=1)
            c_norm = F.normalize(centroids, dim=1)
            cos = f_norm @ c_norm.t()
            mask_other = torch.ones_like(cos).scatter_(
                1, y_hat.unsqueeze(1), 0)
            loss_disperse = (cos.clamp(min=0) * mask_other).sum(1).mean()

            # Symmetric weak-strong consistency
            loss_cons = 0.5 * (kl_with_temperature(z_w, z_s, T=CFG.temp_cons)
                               + kl_with_temperature(z_s, z_w, T=CFG.temp_cons))

            loss = (lambda_attract * loss_attract
                    + lambda_disperse * loss_disperse
                    + lambda_cons * loss_cons)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            with torch.no_grad():
                for c_ in range(CFG.num_classes):
                    idx = (y_hat == c_)
                    if idx.sum() == 0:
                        continue
                    fm = f_w[idx].mean(dim=0)
                    if not initialized[c_]:
                        centroids[c_] = fm
                        initialized[c_] = True
                    else:
                        centroids[c_] = 0.9 * centroids[c_] + 0.1 * fm
                    counts[c_] += idx.sum()

            if verbose:
                pbar.set_postfix(A=f"{loss_attract.item():.3f}",
                                 D=f"{loss_disperse.item():.3f}",
                                 C=f"{loss_cons.item():.3f}")

        accepts.append(1.0)

    return save_results(student, "aad", accepts)