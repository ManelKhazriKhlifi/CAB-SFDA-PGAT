"""dac.py — DaC (Zhang et al., NeurIPS 2022)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, similarity_threshold=0.5, lambda_local=0.5):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    optim   = make_optimizer(student, lr)

    # Source similarity via classifier weight directions
    source_w = F.normalize(student.classifier.weight.detach().clone(), dim=1)

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"DaC {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, f_w = student(xw, return_feat=True)
                z_s, f_s = student(xs, return_feat=True)
            z_w = z_w.float(); f_w = f_w.float()
            z_s = z_s.float(); f_s = f_s.float()

            loss_im = info_max_loss(z_w)

            # Source similarity via cosine
            f_norm = F.normalize(f_w, dim=1)
            cos_k = f_norm @ source_w.t()
            max_cos = cos_k.max(dim=1).values
            src_like = max_cos >= similarity_threshold

            with torch.no_grad():
                p_w = torch.softmax(z_w, dim=1)
                conf, y_hat = p_w.max(dim=1)
                mask = conf >= CFG.conf_thresh_init

            # Broad-class CE on all masked samples
            if mask.sum() > 0:
                loss_pl = F.cross_entropy(z_s[mask], y_hat[mask])
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            # Local contrastive on target-unique
            idx_local = (~src_like) & mask
            if idx_local.sum() > 1:
                fq = F.normalize(f_s[idx_local], dim=1)
                yq = y_hat[idx_local]
                logits_con = fq @ fq.t() / 0.1
                loss_local = F.cross_entropy(logits_con, yq)
            else:
                loss_local = torch.tensor(0.0, device=DEVICE)

            loss = (CFG.lambda_im * loss_im
                    + CFG.lambda_pl * loss_pl
                    + lambda_local * loss_local)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}",
                                 Loc=f"{int(idx_local.sum())}/{mask.sum()}")

        accepts.append(1.0)

    return save_results(student, "dac", accepts)