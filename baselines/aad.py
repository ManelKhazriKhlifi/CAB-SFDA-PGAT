"""aad.py - AaD (Yang et al., NeurIPS 2022)"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, DEVICE
from _common import load_student, make_optimizer, save_results, RunLogger


@torch.no_grad()
def _init_banks(student, dataset, amp_on):
    """Fill the feature/score banks with one pass over the target training set."""
    n = len(dataset)
    fea_bank   = torch.zeros(n, CFG.feat_dim, device=DEVICE)
    score_bank = torch.zeros(n, CFG.num_classes, device=DEVICE)
    loader = DataLoader(dataset, batch_size=CFG.sf_batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    student.eval()
    for xw, _, idx in loader:
        with torch.cuda.amp.autocast(enabled=amp_on):
            z, f = student(xw.to(DEVICE), return_feat=True)
        idx = idx.to(DEVICE)
        fea_bank[idx]   = F.normalize(f.float(), dim=1)
        score_bank[idx] = F.softmax(z.float(), dim=1)
    return fea_bank, score_bank


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, k=3, alpha=1.0, beta=0.75, backbone_lr_mult=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    if backbone_lr_mult == 1.0:
        optim = make_optimizer(student, lr)
    else:   # official AaD: backbone at lr x 0.1, bottleneck at lr
        optim = torch.optim.Adam([
            {"params": student.features.parameters(), "lr": lr * backbone_lr_mult},
            {"params": student.feature_head.parameters(), "lr": lr}])
    logger  = RunLogger("aad", src_ckpt_path,
                        {"arch": arch, "epochs": epochs, "lr": lr,
                         "k": k, "alpha": alpha, "beta": beta,
                         "backbone_lr_mult": backbone_lr_mult})

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    fea_bank, score_bank = _init_banks(student, loaders["tgt_unl"].dataset, amp_on)
    max_iter = epochs * len(loaders["tgt_unl"])
    it = 0

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"AaD {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, _, idx in pbar:
            xw, idx = xw.to(DEVICE), idx.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z, f = student(xw, return_feat=True)
            p = F.softmax(z.float(), dim=1)
            w = alpha * (1 + 10 * it / max_iter) ** (-beta)

            # Update banks, then find the K nearest neighbours (excluding self)
            with torch.no_grad():
                f_norm = F.normalize(f.float(), dim=1)
                fea_bank[idx]   = f_norm
                score_bank[idx] = p.detach()
                kk = min(k, fea_bank.size(0) - 1)
                _, idx_near = (f_norm @ fea_bank.t()).topk(kk + 1, dim=1)
                score_near = score_bank[idx_near[:, 1:]]          # B x K x C

            # Attraction: agree with neighbours' predictions
            loss_attract = -(p.unsqueeze(1) * score_near).sum(-1).sum(1).mean()

            # Dispersion: disagree with the other samples in the batch
            mask = 1.0 - torch.eye(p.size(0), device=DEVICE)
            loss_disperse = ((p @ p.t()) * mask).sum(-1).mean()

            loss = loss_attract + w * loss_disperse

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            it += 1

            logger.step(ep, attract=loss_attract, disperse=loss_disperse, total=loss)
            if verbose:
                pbar.set_postfix(A=f"{loss_attract.item():.3f}",
                                 D=f"{loss_disperse.item():.3f}",
                                 a=f"{w:.3f}")

        accepts.append(1.0)
        logger.end_epoch(ep, accepts[-1], student)

    return save_results(student, "aad", accepts, logger=logger)
