"""
train_source.py
Run this ONCE to produce source_aid_resnet50.pth.
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from config import CFG
from model import ResNetBackbone
from dataloaders import build_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.out_dir, exist_ok=True)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)


def main():
    (src_train_loader, src_test_loader, _, _, _) = build_loaders()
    model = ResNetBackbone(num_classes=CFG.num_classes,
                           feat_dim=CFG.feat_dim,
                           backbone_name=CFG.backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.src_lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, CFG.src_epochs + 1):
        model.train()
        pbar = tqdm(src_train_loader, desc=f"Source {ep}/{CFG.src_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {ep}: src acc = {accuracy(model, src_test_loader):.2f}%")

    ckpt = os.path.join(CFG.out_dir, "source_aid_resnet50.pth")
    torch.save(model.state_dict(), ckpt)
    print("Saved source checkpoint:", ckpt)


if __name__ == "__main__":
    main()
