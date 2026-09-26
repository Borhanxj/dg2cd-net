"""Train DG2CD-Net for one source domain and one seed.

    python scripts/train.py --source photo --seed 0
    python scripts/train.py --source sketch --seed 1 --resume
    python scripts/train.py --source photo --set episodic.batch_size=64 episodic.lr=0.005

Outputs go to runs/<dataset>/<source>/seed<seed>/ unless --output-dir is given:
    config.yaml               exact config used, overrides included
    history.json              per-round, per-episode log
    checkpoints/round_XX.pt   theta_global after each round (resumable)
    theta_global_final.pt    final trainable weights
"""

import argparse
from pathlib import Path

from dg2cd.data import available_synthetic_domains, synthetic_root
from dg2cd.models import DINOViTBackbone
from dg2cd.training import run_episodic_training
from dg2cd.utils import apply_overrides, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DG2CD-Net.")
    p.add_argument("--config", default="configs/pacs.yaml")
    p.add_argument("--source", help="source domain (overrides dataset.source)")
    p.add_argument("--seed", type=int, help="random seed (overrides seed)")
    p.add_argument("--output-dir", help="default: runs/<dataset>/<source>/seed<seed>")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. episodic.lr=0.01")
    p.add_argument("--quiet", action="store_true", help="hide per-epoch loss lines")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.dataset.source = args.source
    if args.seed is not None:
        cfg.seed = args.seed
    apply_overrides(cfg, args.overrides)

    if cfg.dataset.source not in cfg.dataset.domains:
        raise SystemExit(f"unknown source {cfg.dataset.source!r}; "
                         f"choose from {list(cfg.dataset.domains)}")

    output_dir = Path(args.output_dir or
                      Path("runs") / cfg.dataset.name / cfg.dataset.source / f"seed{cfg.seed}")
    cfg.output_dir = str(output_dir)

    # Fail in seconds, not after loading the model, if domains are missing.
    needed = set(cfg.synthetic.train_domains) | set(cfg.synthetic.valid_domains)
    missing = needed - set(available_synthetic_domains(synthetic_root(cfg)))
    if missing:
        raise SystemExit(
            f"missing synthetic domains for source {cfg.dataset.source!r}: {sorted(missing)}\n"
            f"run: python scripts/generate_synthetic_domains.py --source {cfg.dataset.source}"
        )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    ec = cfg.episodic
    print(f"source  : {cfg.dataset.source}   seed: {cfg.seed}   device: {device}")
    print(f"schedule: {ec.n_global} rounds x {ec.n_episodes} episodes x "
          f"{ec.local_epochs} epochs, batch {ec.batch_size}, lr {ec.lr}")
    print(f"output  : {output_dir}")

    encoder = DINOViTBackbone.from_config(cfg.backbone)
    run_episodic_training(cfg, encoder, device, output_dir=output_dir,
                          resume=args.resume, verbose=not args.quiet)

    print(f"\ndone -> {output_dir / 'theta_global_final.pt'}")


if __name__ == "__main__":
    main()