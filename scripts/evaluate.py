"""Evaluate DG2CD-Net on every unseen target domain.

    python scripts/evaluate.py --run runs/pacs/photo/seed0                 # paper protocol
    python scripts/evaluate.py --run runs/pacs/photo/seed0 --k gt          # true K, fast
    python scripts/evaluate.py --run runs/pacs/photo/seed0 --targets cartoon --k-max 100
    python scripts/evaluate.py --baseline --source photo --k gt            # untrained DINO

For each target domain (every domain except the source, unless --targets):
    embed target -> choose K -> K-means -> Hungarian -> All / Old / New
and, for reference, the same at the true class count.

Results are printed as a table and written to <run>/results_k-<mode>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dg2cd.data import PACSDataset, build_transforms, known_novel_split
from dg2cd.eval import (
    estimate_k,
    evaluate_clustering,
    extract_embeddings,
    old_class_indices,
)
from dg2cd.models import DINOViTBackbone
from dg2cd.training import load_params
from dg2cd.utils import load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DG2CD-Net on unseen target domains.")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--run", help="training run directory (contains config.yaml)")
    what.add_argument("--baseline", action="store_true",
                      help="evaluate untrained DINO instead of a trained run")
    p.add_argument("--config", default="configs/pacs.yaml", help="config for --baseline")
    p.add_argument("--source", help="source domain for --baseline")
    p.add_argument("--checkpoint",
                   help="weights to load; default <run>/theta_global_final.pt. "
                        "A round checkpoint (checkpoints/round_XX.pt) also works.")
    p.add_argument("--targets", nargs="*", help="default: every domain except the source")
    p.add_argument("--k", default="estimate",
                   help="'estimate' (Brent, paper protocol), 'gt' (true K), or an integer")
    p.add_argument("--k-max", type=int, help="upper bound for the K search (default eval.k_max)")
    return p.parse_args()


def load_encoder(cfg, args, run_dir: Path) -> tuple[DINOViTBackbone, str]:
    """DINO backbone, with trained block-11 weights applied unless --baseline."""
    encoder = DINOViTBackbone.from_config(cfg.backbone)
    if args.baseline:
        return encoder, "DINO (untrained)"

    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "theta_global_final.pt"
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    if "theta_global" in state:            # round checkpoint: unwrap
        state = state["theta_global"]
    load_params(encoder, state)
    return encoder, str(ckpt)


def main() -> None:
    args = parse_args()

    # --- config and output location ---------------------------------------
    if args.baseline:
        cfg = load_config(args.config)
        if args.source:
            cfg.dataset.source = args.source
        out_dir = Path("runs") / cfg.dataset.name / cfg.dataset.source / "baseline"
    else:
        out_dir = Path(args.run)
        cfg = load_config(out_dir / "config.yaml")     # exactly what was trained
    out_dir.mkdir(parents=True, exist_ok=True)

    k_mode = args.k if args.k in ("estimate", "gt") else int(args.k)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    known, _ = known_novel_split(cfg.dataset)
    source = cfg.dataset.source
    targets = args.targets or [d for d in cfg.dataset.domains if d != source]
    if source in targets:
        raise SystemExit(f"target list contains the source domain {source!r}")

    encoder, weights_desc = load_encoder(cfg, args, out_dir)
    _, eval_tf = build_transforms(encoder.data_config(), cfg.augmentation)

    print(f"source : {source}    weights: {weights_desc}")
    print(f"targets: {targets}    K: {k_mode}\n")

    # --- labelled source embeddings: only needed to estimate K ------------
    if k_mode == "estimate":
        src_ds = PACSDataset(cfg.dataset.root, source, known, transform=eval_tf)
        src_feats, src_labels = extract_embeddings(encoder, src_ds, device)
        k_min = cfg.eval.k_min or len(known)
        k_max = args.k_max or cfg.eval.k_max

    # --- evaluate each target ---------------------------------------------
    results: dict[str, dict] = {}
    for target in targets:
        tgt_ds = PACSDataset(cfg.dataset.root, target, cfg.dataset.class_names, transform=eval_tf)
        feats, labels = extract_embeddings(encoder, tgt_ds, device)
        old_ids = old_class_indices(tgt_ds.classes, known)
        gt_k = len(tgt_ds.classes)

        k_history = None
        if k_mode == "estimate":
            print(f"[{target}] estimating K in [{k_min}, {k_max}] ...")
            k, k_history = estimate_k(
                src_feats, src_labels, feats, k_min=k_min, k_max=k_max,
                seed=cfg.seed, n_init=cfg.eval.k_estimation_n_init, verbose=False,
            )
        elif k_mode == "gt":
            k = gt_k
        else:
            k = k_mode

        res = evaluate_clustering(feats, labels, old_ids, k=k,
                                  seed=cfg.seed, n_init=cfg.eval.kmeans_n_init)
        at_gt = res if k == gt_k else evaluate_clustering(
            feats, labels, old_ids, k=gt_k, seed=cfg.seed, n_init=cfg.eval.kmeans_n_init)

        results[target] = {
            "k": k,
            "all": res["all"], "old": res["old"], "new": res["new"],
            "gt_k": gt_k,
            "at_gt_k": {"all": at_gt["all"], "old": at_gt["old"], "new": at_gt["new"]},
            "k_history": k_history,
        }

    # --- report --------------------------------------------------------------
    print(f"\n{'target':<14}{'K':>5}{'All':>9}{'Old':>9}{'New':>9}   | All@K=gt")
    print("-" * 60)
    for t, r in results.items():
        print(f"{t:<14}{r['k']:>5}{r['all']:>9.4f}{r['old']:>9.4f}{r['new']:>9.4f}"
              f"   | {r['at_gt_k']['all']:.4f}")
    mean = {m: float(np.mean([r[m] for r in results.values()])) for m in ("all", "old", "new")}
    mean_gt = float(np.mean([r["at_gt_k"]["all"] for r in results.values()]))
    print("-" * 60)
    print(f"{'mean':<14}{'':>5}{mean['all']:>9.4f}{mean['old']:>9.4f}{mean['new']:>9.4f}"
          f"   | {mean_gt:.4f}")

    out_path = out_dir / f"results_k-{k_mode}.json"
    out_path.write_text(json.dumps({
        "source": source,
        "weights": weights_desc,
        "k_mode": k_mode,
        "seed": cfg.seed,
        "targets": results,
        "mean": mean,
        "mean_all_at_gt_k": mean_gt,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()