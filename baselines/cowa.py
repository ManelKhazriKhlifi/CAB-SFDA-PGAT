"""cowa.py — CoWA (Lee et al., ICML 2022)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, w_predictive=0.5, w_structure=0.5):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    optim   = make_optimizer(student, lr)

    # Running per-class feature centroids
    centroids   = torch.zeros(CFG.num_classes, CFG.feat_dim, device=DEVICE)
    counts      = torch.zeros(CFG.num_classes, device=DEVICE)
    initialized = torch.zeros(CFG.num_classes, dtype=torch.bool, device=DEVICE)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"CoWA {CFG.task} s{CFG.seed} ep {ep}/{epochs}")
        acc, obs = 0, 0

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, f_w = student(xw, return_feat=True)
                z_s, _   = student(xs, return_feat=True)
            z_w = z_w.float(); f_w = f_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            p_w = torch.softmax(z_w, dim=1)
            conf, y_hat = p_w.max(dim=1)

            # Structural confidence: cosine to class centroid
            f_norm = F.normalize(f_w, dim=1)
            c_norm = F.normalize(centroids, dim=1)
            cos = f_norm @ c_norm.t()
            cos = ((1.0 + cos) / 2.0).clamp(0, 1)
            struct = cos.gather(1, y_hat.unsqueeze(1)).squeeze(1)
            struct[~initialized[y_hat]] = conf[~initialized[y_hat]]

            score = w_predictive * conf + w_structure * struct
            mask  = score >= CFG.conf_thresh_init

            obs += mask.numel(); acc += int(mask.sum().item())

            if mask.sum() > 0:
                loss_pl = F.cross_entropy(z_s[mask], y_hat[mask])
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            loss = CFG.lambda_im * loss_im + CFG.lambda_pl * loss_pl
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            with torch.no_grad():
                for c in range(CFG.num_classes):
                    idx = (y_hat == c) & mask
                    if idx.sum() == 0:
                        continue
                    fm = f_w[idx].mean(dim=0)
                    if not initialized[c]:
                        centroids[c] = fm
                        initialized[c] = True
                    else:
                        centroids[c] = 0.9 * centroids[c] + 0.1 * fm
                    counts[c] += idx.sum()

            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}",
                                 Acc=f"{100*acc/max(obs,1):.1f}%")

        accepts.append(acc / max(obs, 1))

    return save_results(student, "cowa", accepts)