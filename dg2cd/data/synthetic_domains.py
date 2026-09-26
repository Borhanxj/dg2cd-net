"""Synthetic domain generation for PACS dataset."""

from collections.abc import Sequence
from pathlib import Path
from PIL import Image
from numpy import rec
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

FALLBACK_RECIPES: dict[str, dict[str, float]] = {
    # the 6 training domains 
    "rainy":  dict(hue=-0.08, brightness=0.75, contrast=0.85, saturation=0.50, blur=1.2),
    "snowy":   dict(hue=-0.03, brightness=1.30, contrast=0.75, saturation=0.40, blur=0.6),
    "white":  dict(hue=0.00,  brightness=1.45, contrast=0.70, saturation=0.25, blur=0.0),
    "black":  dict(hue=0.00,  brightness=0.45, contrast=1.15, saturation=0.60, blur=0.0),
    "forest": dict(hue=0.10,  brightness=0.85, contrast=1.00, saturation=1.35, blur=0.0),
    "beach":  dict(hue=0.05,  brightness=1.15, contrast=1.05, saturation=1.30, blur=0.0),
    # 3 validation domains
    "summer": dict(hue=0.02,  brightness=1.20, contrast=1.00, saturation=1.45, blur=0.0),
    "gray":   dict(hue=0.00,  brightness=1.00, contrast=1.00, saturation=0.00, blur=0.0),
    "urban":  dict(hue=-0.05, brightness=0.90, contrast=1.25, saturation=0.65, blur=0.0),
}

def synthetic_root(cfg) -> Path:
    """Cache directory for the CURRENT source's synthetic domains.

    Synthetic domains are restyled copies of the source domain, so each
    source needs its own set. Sharing one cache across sources would feed,
    e.g., restyled photos to a sketch-source run.
    """
    return Path(cfg.synthetic.cache_dir) / cfg.dataset.source

def _apply_recipe(image: Image.Image, recipe: dict[str, float]) -> Image.Image:
    """Photometric transformation of an image according to a recipe of parameters."""
    out = image
    if recipe.get("brightness", 1.0) != 1.0:
        out = TF.adjust_brightness(out, recipe["brightness"])
    if recipe.get("contrast", 1.0) != 1.0:
        out = TF.adjust_contrast(out, recipe["contrast"])
    if recipe.get("saturation", 1.0) != 1.0:
        out = TF.adjust_saturation(out, recipe["saturation"])
    if recipe.get("hue", 0.0) != 0.0:
        out = TF.adjust_hue(out, recipe["hue"])
    if recipe.get("blur", 0.0) > 0.0:
        out = TF.gaussian_blur(out, kernel_size=5, sigma=recipe["blur"])
    return out

def generate_synthetic_domain(
    source_root: str | Path,
    source_domain: str,
    classes: Sequence[str],
    cache_dir: str | Path,
    domain_name: str,
    generator: str = "augmentation",
    overwrite: bool = False,
) -> int:
    """Generate one synthetic domain from the source domain.

    Args:
        source_root: PACS root (contains the domain folders).
        source_domain: domain to restyle, e.g. "photo".
        classes: known classes Y_s -- the only ones present in D_syn.
        cache_dir: output root, e.g. "data/PACS_synth".
        domain_name: one of FALLBACK_RECIPES, e.g. "rainy".
        generator: "augmentation" | "instructpix2pix".
        overwrite: regenerate images that already exist.

    Returns:
        Number of images written (skipped ones are not counted).
    """
    if generator == "instructpix2pix":
        raise NotImplementedError(
            "InstructPix2Pix generator is not implemented yet."
        )
    if generator != "augmentation":
        raise ValueError(f"unknown generator: {generator!r}")
    if domain_name not in FALLBACK_RECIPES:
        raise KeyError(
            f"no fallback recipe for {domain_name!r}; "
            f"known: {sorted(FALLBACK_RECIPES)}"
        )

    recipe = FALLBACK_RECIPES[domain_name]
    source_root, cache_dir = Path(source_root), Path(cache_dir)
    written = 0

    for class_name in classes:
        in_dir = source_root / source_domain / class_name
        if not in_dir.is_dir():
            raise FileNotFoundError(f"source class folder not found: {in_dir}")

        out_dir = cache_dir / domain_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_path in sorted(p for p in in_dir.iterdir()
                               if p.suffix.lower() in IMAGE_EXTENSIONS):
            dst_path = out_dir / f"{src_path.stem}.jpg"
            if dst_path.exists() and not overwrite:
                continue
            with Image.open(src_path) as img:
                out = _apply_recipe(img.convert("RGB"), recipe)
                out.save(dst_path, quality=95)
            written += 1

    return written

def generate_all_domains(cfg, overwrite: bool = False) -> dict[str, int]:
    """Generate every training and validation domain listed in the config."""
    from .datasets import known_novel_split

    known, _ = known_novel_split(cfg.dataset)
    domains = list(cfg.synthetic.train_domains) + list(cfg.synthetic.valid_domains)

    counts = {}
    for name in domains:
        counts[name] = generate_synthetic_domain(
            source_root=cfg.dataset.root,
            source_domain=cfg.dataset.source,
            classes=known,
            cache_dir=synthetic_root(cfg),
            domain_name=name,
            generator=cfg.synthetic.generator,
            overwrite=overwrite,
        )
    return counts

def available_synthetic_domains(cache_dir: str | Path) -> list[str]:
    """List the synthetic domains that have been generated."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    return sorted(p.name for p in cache_dir.iterdir() if p.is_dir())