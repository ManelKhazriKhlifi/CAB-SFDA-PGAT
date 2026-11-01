"""nrc.py — NRC (Yang et al., NeurIPS 2021)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, k=5):
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
                    desc=f"NRC {CFG.task} s{CFG.seed} ep {ep}/{epochs}")
        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, f_w = student(xw, return_feat=True)
                z_s, _   = student(xs, return_feat=True)
            z_w = z_w.float(); f_w = f_w.float(); z_s = z_s.float()

            loss_im = info_max_loss(z_w)

            with torch.no_grad():
                y_hat = z_w.argmax(dim=1)

            # K-NN refinement of pseudo-labels
            f_norm = F.normalize(f_w, dim=1)
            sim = f_norm @ f_norm.t()
            sim.fill_diagonal_(-1.0)
            kk = min(k, f_norm.size(0) - 1)
            knn_w, knn_i = sim.topk(kk, dim=1)

            with torch.no_grad():
                neigh_labels = y_hat[knn_i]
                votes = torch.zeros(f_norm.size(0), CFG.num_classes, device=DEVICE)
                votes.scatter_add_(1, neigh_labels, knn_w.clamp(min=0))
                y_ref = votes.argmax(dim=1)

            loss_pl = F.cross_entropy(z_s, y_ref)
            loss = CFG.lambda_im * loss_im + CFG.lambda_pl * loss_pl

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}")

        accepts.append(1.0)

    return save_results(student, "nrc", accepts)