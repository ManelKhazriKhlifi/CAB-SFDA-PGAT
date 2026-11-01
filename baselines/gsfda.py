"""gsfda.py — G-SFDA (Yang et al., ICCV 2021)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from _common import load_student, make_optimizer, save_results


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, k=3, lambda_cluster=0.5):
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
                    desc=f"G-SFDA {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z, f = student(xw, return_feat=True)
            z = z.float(); f = f.float()

            p = torch.softmax(z, dim=1)
            loss_ent = -(p * (p + 1e-8).log()).sum(dim=1).mean()

            # Local structure clustering
            f_norm = F.normalize(f, dim=1)
            sim = f_norm @ f_norm.t()
            sim.fill_diagonal_(-1.0)
            kk = min(k, f_norm.size(0) - 1)
            knn_w, knn_i = sim.topk(kk, dim=1)
            knn_w = F.softmax(knn_w / 0.1, dim=1)
            neigh_f = f[knn_i]
            weighted = (knn_w.unsqueeze(-1) * neigh_f).sum(dim=1)
            loss_cluster = F.mse_loss(f, weighted)

            loss_reg = sum((p_ ** 2).mean()
                           for p_ in student.classifier.parameters())

            loss = loss_ent + lambda_cluster * loss_cluster + 0.01 * loss_reg
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            if verbose:
                pbar.set_postfix(Ent=f"{loss_ent.item():.3f}",
                                 Clu=f"{loss_cluster.item():.3f}")

        accepts.append(1.0)

    return save_results(student, "gsfda", accepts)