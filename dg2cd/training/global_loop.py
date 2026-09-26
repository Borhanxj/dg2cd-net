"""Episodic training with validation-weighted task-vector merging.

One encoder object is reused throughout. Only the trainable parameters
(block 11) ever change, so "copy theta_global into theta_local" is just
writing a 12-tensor snapshot back into the model -- no deepcopy of 86M
parameters per episode.
"""

import json
import time
from pathlib import Path

import torch
import yaml

from ..data import (
    Episode,
    EpisodeSampler,
    build_transforms,
    build_validation_dataset,
    known_novel_split,
)
from ..eval import evaluate_clustering, extract_embeddings, old_class_indices
from .local_episode import train_local_episode
from .task_vectors import (
    load_params,
    merge,
    snapshot,
    softmax_weights,
    state_norm,
    task_vector,
    trainable_names,
)


def validate_episode(encoder, valid_dataset, known: list[str], episode: Episode,
                     device: torch.device, cfg) -> dict[str, float]:
    """GCD accuracy of theta_local on D_valid.

    K is the ground-truth |Y_s|: D_valid is built from the known classes, so
    the class count is known during training. Estimating K here would add
    noise unrelated to which episode trained the better embedding.
    """
    features, labels = extract_embeddings(encoder, valid_dataset, device)
    old_ids = old_class_indices(known, episode.known_classes)
    return evaluate_clustering(
        features, labels, old_ids,
        k=len(known), seed=cfg.seed, n_init=cfg.eval.kmeans_n_init,
    )


def _save_checkpoint(path: Path, round_idx: int, theta_global, sampler, history) -> None:
    """Save a checkpoint of the global model and training state for resuming later."""
    torch.save(
        {
            "round": round_idx,
            "theta_global": theta_global,
            "sampler_rng": sampler.rng.getstate(),
            "history": history,
        },
        path,
    )


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("round_*.pt"))
    return files[-1] if files else None


def run_episodic_training(
    cfg,
    encoder,
    device: torch.device,
    output_dir: str | Path | None = None,
    resume: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Run episodic training and return the per-round history.

    Args:
        cfg: full config.
        encoder: DINOViTBackbone holding theta_global^0 (DINO weights).
            Updated in place; holds theta_global^{ng} on return.
        device: training device.
        output_dir: where checkpoints, history.json and the config copy go.
            Defaults to cfg.output_dir.
        resume: continue from the latest round checkpoint in output_dir.
        verbose: per-epoch loss lines from the local episodes.

    Returns:
        One dict per global round (also written to history.json).
    """
    output_dir = Path(output_dir or cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )

    encoder.to(device)
    known, _ = known_novel_split(cfg.dataset)
    train_tf, eval_tf = build_transforms(encoder.data_config(), cfg.augmentation)
    sampler = EpisodeSampler(cfg, transform=train_tf)
    valid_dataset, _ = build_validation_dataset(cfg, transform=eval_tf)
    names = trainable_names(encoder)

    history: list[dict] = []
    start_round = 1

    if resume:
        ckpt = _latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location="cpu")
            load_params(encoder, state["theta_global"])
            sampler.rng.setstate(state["sampler_rng"])
            history = state["history"]
            start_round = state["round"] + 1
            print(f"resuming from {ckpt.name} -> starting at round {start_round}")

    for g in range(start_round, cfg.episodic.n_global + 1):
        round_start = time.perf_counter()
        theta_global = snapshot(encoder, names)
        episodes = sampler.sample_global_round(g)

        print(f"\n=== global round {g}/{cfg.episodic.n_global} ===")
        deltas, scores, episode_logs = [], [], []

        for ep in episodes:
            ep_start = time.perf_counter()
            load_params(encoder, theta_global)

            losses = train_local_episode(encoder, ep, cfg, device, verbose=verbose)

            theta_local = snapshot(encoder, names)
            delta = task_vector(theta_global, theta_local)
            deltas.append(delta)

            val = validate_episode(encoder, valid_dataset, known, ep, device, cfg)
            scores.append(val["all"])

            log = {
                "episode": ep.episode_index,
                "synthetic_domain": ep.synthetic_domain,
                "known_classes": ep.known_classes,
                "novel_classes": ep.novel_classes,
                "final_losses": {k: v[-1] for k, v in losses.items()},
                "valid_all": val["all"],
                "valid_old": val["old"],
                "valid_new": val["new"],
                "delta_norm": state_norm(delta),
                "seconds": round(time.perf_counter() - ep_start, 1),
            }
            episode_logs.append(log)
            print(
                f"  e{ep.episode_index} [{ep.synthetic_domain:<6}] "
                f"Y_s^eg={ep.known_classes}  "
                f"valid All {val['all']:.4f} | Old {val['old']:.4f} | New {val['new']:.4f}  "
                f"||delta|| {log['delta_norm']:.3f}  ({log['seconds']}s)"
            )

        weights = softmax_weights(scores)
        theta_new = merge(theta_global, deltas, weights)
        load_params(encoder, theta_new)

        update_norm = state_norm({n: theta_new[n] - theta_global[n] for n in names})
        round_log = {
            "round": g,
            "episodes": episode_logs,
            "weights": weights.tolist(),
            "update_norm": update_norm,
            "seconds": round(time.perf_counter() - round_start, 1),
        }
        history.append(round_log)

        _save_checkpoint(ckpt_dir / f"round_{g:02d}.pt", g, theta_new, sampler, history)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"  weights {[round(w, 3) for w in weights.tolist()]}  "
            f"||theta^g - theta^(g-1)|| {update_norm:.4f}  ({round_log['seconds']}s)"
        )

    torch.save(snapshot(encoder, names), output_dir / "theta_global_final.pt")
    return history