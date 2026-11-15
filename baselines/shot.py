
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results, RunLogger


@torch.no_grad()
def _cluster_labels(student, dataset, amp_on):
    """SHOT obtain_label: centroid-based pseudo-labels for every target sample."""
    loader = DataLoader(dataset, batch_size=CFG.sf_batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    feats  = torch.zeros(len(dataset), CFG.feat_dim + 1, device=DEVICE)
    probs  = torch.zeros(len(dataset), CFG.num_classes, device=DEVICE)
    student.eval()
    for xw, _, idx in loader:
        with torch.cuda.amp.autocast(enabled=amp_on):
            z, f = student(xw.to(DEVICE), return_feat=True)
        idx = idx.to(DEVICE)
        f = torch.cat([f.float(), torch.ones(f.size(0), 1, device=DEVICE)], dim=1)
        feats[idx] = F.normalize(f, dim=1)
        probs[idx] = F.softmax(z.float(), dim=1)

    aff = probs
    for _ in range(2):                      # soft centroids, then one hard refinement
        centroids = aff.t() @ feats / (aff.sum(dim=0)[:, None] + 1e-8)
        labels = (feats @ F.normalize(centroids, dim=1).t()).argmax(dim=1)
        aff = F.one_hot(labels, CFG.num_classes).float()
    agree = (labels == probs.argmax(dim=1)).float().mean().item()
    return labels, agree


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None, verbose=True,
        cls_par=0.3, backbone_lr_mult=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    if backbone_lr_mult == 1.0:
        optim = make_optimizer(student, lr)
    else:   # official SHOT: backbone at lr x 0.1, bottleneck at lr
        optim = torch.optim.Adam([
            {"params": student.features.parameters(), "lr": lr * backbone_lr_mult},
            {"params": student.feature_head.parameters(), "lr": lr}])
    logger  = RunLogger("shot", src_ckpt_path,
                        {"arch": arch, "epochs": epochs, "lr": lr, "cls_par": cls_par,
                         "backbone_lr_mult": backbone_lr_mult})

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        
        pseudo, agree = _cluster_labels(student, loaders["tgt_unl"].dataset, amp_on)
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"SHOT {CFG.task} s{CFG.seed} ep {ep}/{epochs}")
        for xw, _, idx in pbar:
            xw = xw.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z, _ = student(xw, return_feat=True)
            z = z.float()

            loss_im = info_max_loss(z)
            loss_pl = F.cross_entropy(z, pseudo[idx.to(DEVICE)])

            loss = loss_im + cls_par * loss_pl
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            logger.step(ep, im=loss_im, pl=loss_pl, total=loss)
            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}")

        accepts.append(agree)
        logger.end_epoch(ep, accepts[-1], student)

    return save_results(student, "shot", accepts, logger=logger)
