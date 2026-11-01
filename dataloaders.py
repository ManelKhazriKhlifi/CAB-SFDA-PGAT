import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from config import CFG


IMG_SIZE      = CFG.img_size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


src_train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
src_test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Weak augmentation: resize + horizontal flip
tgt_weak_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Strong augmentation: crop+padding, affine (rot/trans/scale/shear),
# color jitter (brightness/contrast/saturation/hue), Gaussian blur
tgt_strong_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomCrop(IMG_SIZE, padding=8),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                            scale=(0.9, 1.1), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TwoViewTarget(Dataset):
    """Returns (weak_view, strong_view, dummy_label)."""
    def __init__(self, base_ds, weak_tf, strong_tf):
        self.base = base_ds
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.weak_tf(img), self.strong_tf(img), -1


def build_loaders():
    """Build all loaders for the current task."""
    source_root = f"{CFG.data_root}/data/source_{CFG.source_name.lower()}_split"
    target_root = f"{CFG.data_root}/data/target_{CFG.target_name.lower()}_split"

    src_train     = torchvision.datasets.ImageFolder(f"{source_root}/train", transform=src_train_tf)
    src_test      = torchvision.datasets.ImageFolder(f"{source_root}/test",  transform=src_test_tf)
    tgt_train_raw = torchvision.datasets.ImageFolder(f"{target_root}/train", transform=None)
    tgt_test      = torchvision.datasets.ImageFolder(f"{target_root}/test",  transform=src_test_tf)

    assert src_train.classes == tgt_train_raw.classes, \
        "Source/target class sets must match"
    assert len(src_train.classes) == CFG.num_classes, \
        f"Expected {CFG.num_classes} classes, got {len(src_train.classes)}"

    tgt_train = TwoViewTarget(tgt_train_raw, tgt_weak_tf, tgt_strong_tf)

    return {
        "src_train":    DataLoader(src_train, batch_size=CFG.src_batch_size,
                                   shuffle=True,  num_workers=CFG.num_workers,
                                   pin_memory=CFG.pin_memory),
        "src_test":     DataLoader(src_test,  batch_size=CFG.src_batch_size,
                                   shuffle=False, num_workers=CFG.num_workers,
                                   pin_memory=CFG.pin_memory),
        "tgt_unl":      DataLoader(tgt_train, batch_size=CFG.sf_batch_size,
                                   shuffle=True,  num_workers=CFG.num_workers,
                                   pin_memory=CFG.pin_memory),
        "tgt_test":     DataLoader(tgt_test,  batch_size=CFG.sf_batch_size,
                                   shuffle=False, num_workers=CFG.num_workers,
                                   pin_memory=CFG.pin_memory),
        "tgt_train_raw": tgt_train_raw,
        "class_names":  src_train.classes,
    }