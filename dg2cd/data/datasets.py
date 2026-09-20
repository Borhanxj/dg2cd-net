"""PACS dataset and image transforms."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class PACSDataset(Dataset):
    """Images from one PACS domain, restricted to a given class list.
    Args:
        root: directory containing the domain folders.
        domain: one of art_painting / cartoon / photo / sketch.
        classes: ordered class names to include. Labels are indices into this
            list, so the ordering is part of the contract -- keep it stable.
        transform: applied to the PIL image. May return a tuple (see
            TwoViewTransform), in which case __getitem__ returns that tuple.
    """
    def __init__(
        self,
        root: str | Path,
        domain: str,
        classes: Sequence[str],
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.domain = domain
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform

        domain_dir = self.root / self.domain
        if not domain_dir.is_dir():
            raise ValueError(f"Domain directory {domain_dir} does not exist.")

        # list of images and teir corresponding class indices
        self.samples: list[tuple[Path, int]] = []

        for name in self.classes:
            class_dir = domain_dir / name
            if not class_dir.is_dir():
                raise ValueError(f"Class directory {class_dir} does not exist.")

            files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise ValueError(f"No image files found in {class_dir}.")
            self.samples.extend((p, self.class_to_idx[name]) for p in files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.classes, 0)
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def labels_tensor(self) -> torch.Tensor:
        return torch.tensor([label for _, label in self.samples], dtype=torch.long)

class TwoViewTransform:
    """Produce (x, x+) for the unsupervised contrastive loss

    View 1 is the original image, view 2 is a random augmentation of it.
    """
    def __init__(self, base:Callable, augment:Callable):
        self.base = base
        self.augment = augment

    def __call__(self, x)-> tuple[torch.Tensor, torch.Tensor]:
        return self.base(x), self.augment(x)

def build_transforms(data_config: dict, augmentation_cfg) -> tuple[Callable, Callable]:
    """Build (train, eval) transforms from the backbone's data config.

    Args:
        data_config: DINOViTBackbone.data_config() -- carries the input size,
            mean, std and crop ratio the pretrained weights expect.
        augmentation_cfg: the `augmentation` section of the loaded config.

    Returns:
        train_transform: TwoViewTransform yielding (x, x+).
        eval_transform: deterministic, for validation and target extraction.
    """
    size = data_config["input_size"][-1]
    mean, std = data_config["mean"], data_config["std"]
    crop_pct = data_config.get("crop_pct", 0.9)
    resize = int(round(size / crop_pct))

    # Base transform for both train and eval, resizing and normalizing the image.
    base = T.Compose([
        T.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Augmentation transform for the second view, using the config-specified augmentations.
    geometric = T.Compose([
        T.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.RandomHorizontalFlip(p=augmentation_cfg.horizontal_flip),
        T.RandomAffine(
            degrees=augmentation_cfg.rotation_degrees,
            translate=tuple(augmentation_cfg.translate),
            scale=tuple(augmentation_cfg.scale),
            interpolation=InterpolationMode.BILINEAR,
        ),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return TwoViewTransform(base, geometric), base

def known_novel_split(dataser_cfg) -> tuple[list[str], list[str]]:
    """Split the class vocabulary into Y_s (known) and Y_t^new (novel).

    Order is taken from `class_names`, not from `known_classes`, so both
    lists are stable regardless of how the config lists them.
    """
    all_classes = list(dataser_cfg.class_names)
    known_set = set(dataser_cfg.known_classes)

    unknown = known_set - set(all_classes)
    if unknown:
        raise ValueError(f"Known classes {unknown} are not in the class_names list.")

    known = [c for c in all_classes if c in known_set]
    novel = [c for c in all_classes if c not in known_set]

    if not novel:
        raise ValueError("No novel classes left, known_classes covers everything.")
    
    return known, novel