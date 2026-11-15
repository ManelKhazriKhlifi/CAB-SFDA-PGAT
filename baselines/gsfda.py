import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from _common import load_student, make_optimizer, save_results, RunLogger
from aad import _init_banks


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None,
        verbose=True, k=3, lambda_div=1.0, backbone_lr_mult=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    if backbone_lr_mult == 1.0:
        optim = make_optimizer(student, lr)
    else:   
        optim = torch.optim.Adam([
            {"params": student.features.parameters(), "lr": lr * backbone_lr_mult},
            {"params": student.feature_head.parameters(), "lr": lr}])
    logger  = RunLogger("gsfda", src_ckpt_path,
                        {"arch": arch, "epochs": epochs, "lr": lr, "k": k,
                         "lambda_div": lambda_div,
                         "backbone_lr_mult": backbone_lr_mult})

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    fea_bank, score_bank = _init_banks(student, loaders["tgt_unl"].dataset, amp_on)

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"G-SFDA {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, _, idx in pbar:
            xw, idx = xw.to(DEVICE), idx.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z, f = student(xw, return_feat=True)
            p = F.softmax(z.float(), dim=1)

            with torch.no_grad():
                f_norm = F.normalize(f.float(), dim=1)
                fea_bank[idx]   = f_norm
                score_bank[idx] = p.detach()
                kk = min(k, fea_bank.size(0) - 1)
                _, idx_near = (f_norm @ fea_bank.t()).topk(kk + 1, dim=1)
                score_near = score_bank[idx_near[:, 1:]]          

            dot = (p.unsqueeze(1) * score_near).sum(-1)            
            loss_cluster = -torch.log(dot + 1e-8).sum(1).mean()

            p_mean = p.mean(dim=0)
            loss_div = (p_mean * torch.log(p_mean + 1e-8)).sum()

            loss = loss_cluster + lambda_div * loss_div
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            logger.step(ep, cluster=loss_cluster, div=loss_div, total=loss)
            if verbose:
                pbar.set_postfix(Clu=f"{loss_cluster.item():.3f}",
                                 Div=f"{loss_div.item():.3f}")

        accepts.append(1.0)
        logger.end_epoch(ep, accepts[-1], student)

    return save_results(student, "gsfda", accepts, logger=logger)
