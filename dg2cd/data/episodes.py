"""Episode construction"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .datasets import PACSDataset, known_novel_split
from .synthetic_domains import synthetic_root


@dataclass
class Episode:
    """One episode's data and metadata.

    Attributes:
        global_index: g, 1-based.
        episode_index: e, 1-based.
        known_classes: Y_s^eg -- labels in source_dataset are indices INTO
            this list, so an episode with 3 known classes yields labels 0..2.
        novel_classes: Y_s \\ Y_s^eg -- present unlabelled in D_syn^eg.
        synthetic_domain: name of the chosen pseudo-domain.
        source_dataset: D_S^eg, labelled.
        synthetic_dataset: D_syn^eg. It DOES carry labels (indices into the
            full Y_s), but they are for diagnostics only and must never reach
            a loss -- the whole point is that this domain is unlabelled.
    """

    global_index: int
    episode_index: int
    known_classes: list[str]
    novel_classes: list[str]
    synthetic_domain: str
    source_dataset: Dataset
    synthetic_dataset: Dataset = field(repr=False)

    @property
    def num_known_classes(self) -> int:
        """|Y_s^eg| -- the episode classifier needs this + 1 outputs."""
        return len(self.known_classes)

    def __repr__(self) -> str:
        return (
            f"Episode(g={self.global_index}, e={self.episode_index}, "
            f"Y_s^eg={self.known_classes}, novel={self.novel_classes}, "
            f"D_syn={self.synthetic_domain!r}, "
            f"|D_S^eg|={len(self.source_dataset)}, "
            f"|D_syn^eg|={len(self.synthetic_dataset)})"
        )


class EpisodeSampler:
    """Builds episodes for the episodic training loop.

    Args:
        cfg: full loaded config.
        transform: applied to both source and synthetic images. During
            training this is the TwoViewTransform (x, x+).
        seed: overrides cfg.seed. The sampler owns its own RNG so episode
            composition is reproducible independently of torch/numpy state.
    """

    def __init__(self, cfg, transform: Callable | None = None, seed: int | None = None):
        self.cfg = cfg
        self.transform = transform
        self.rng = random.Random(cfg.seed if seed is None else seed)

        self.known_classes, self.novel_classes = known_novel_split(cfg.dataset)
        self.train_domains = list(cfg.synthetic.train_domains)

        lo, hi = cfg.episodic.episode_known_range
        if not 1 <= lo <= hi:
            raise ValueError(f"episode_known_range must satisfy 1 <= lo <= hi, got {[lo, hi]}")
        if hi >= len(self.known_classes):
            raise ValueError(
                f"episode_known_range upper bound {hi} must be < |Y_s| = "
                f"{len(self.known_classes)}, otherwise an episode can have no "
                f"novel classes in D_syn^eg"
            )
        self.known_range = (lo, hi)

        self.synthetic_root = synthetic_root(cfg)
        missing = [d for d in self.train_domains if not (self.synthetic_root / d).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"synthetic domains not generated for source "
                f"{cfg.dataset.source!r}: {missing}\n"
                f"run scripts/generate_synthetic_domains.py --source {cfg.dataset.source}"
            )

    def domain_order(self, num_episodes: int) -> list[str]:
        """Algorithm 1 line 2: shuffle the synthetic domains for this round.

        With ne == len(train_domains) this is a permutation, so each domain is
        used exactly once per global round. If ne exceeds the number of
        domains, further shuffled passes are appended.
        """
        order: list[str] = []
        while len(order) < num_episodes:
            shuffled = list(self.train_domains)
            self.rng.shuffle(shuffled)
            order.extend(shuffled)
        return order[:num_episodes]

    def _sample_known_subset(self) -> tuple[list[str], list[str]]:
        """Draw Y_s^eg, a proper subset of Y_s."""
        lo, hi = self.known_range
        k = self.rng.randint(lo, hi)
        chosen = set(self.rng.sample(self.known_classes, k))
        # Preserve the canonical ordering so label indices are predictable.
        episode_known = [c for c in self.known_classes if c in chosen]
        episode_novel = [c for c in self.known_classes if c not in chosen]
        return episode_known, episode_novel

    def _subsample(self, dataset: Dataset) -> Dataset:
        """Take D_S^eg as a fraction of the available source images."""
        fraction = self.cfg.episodic.episode_source_fraction
        if fraction >= 1.0:
            return dataset
        n = max(1, int(round(len(dataset) * fraction)))
        indices = self.rng.sample(range(len(dataset)), n)
        return Subset(dataset, sorted(indices))

    def sample_episode(
        self,
        global_index: int,
        episode_index: int,
        synthetic_domain: str,
    ) -> Episode:
        """Build one episode."""
        episode_known, episode_novel = self._sample_known_subset()

        # Labels are indices into episode_known, so the model literally cannot
        # emit a label for an episode-novel class.
        source = PACSDataset(
            root=self.cfg.dataset.root,
            domain=self.cfg.dataset.source,
            classes=episode_known,
            transform=self.transform,
        )

        # D_syn^eg carries the FULL known set -- that is where the episode's
        # novel classes live.
        synthetic = PACSDataset(
            root=self.synthetic_root,
            domain=synthetic_domain,
            classes=self.known_classes,
            transform=self.transform,
        )

        return Episode(
            global_index=global_index,
            episode_index=episode_index,
            known_classes=episode_known,
            novel_classes=episode_novel,
            synthetic_domain=synthetic_domain,
            source_dataset=self._subsample(source),
            synthetic_dataset=synthetic,
        )

    def sample_global_round(self, global_index: int) -> list[Episode]:
        """All ne episodes of one global update."""
        n = self.cfg.episodic.n_episodes
        return [
            self.sample_episode(global_index, e + 1, domain)
            for e, domain in enumerate(self.domain_order(n))
        ]



def build_validation_dataset(
    cfg,
    transform: Callable | None = None,
) -> tuple[Dataset, torch.Tensor]:
    """D_valid -- the held-out synthetic domains, concatenated.

    Built once and reused for every episode. The paper is explicit that a
    consistent, separate validation distribution is what stabilises the
    task-vector weighting; ablation (vi) shows per-episode validation costs
    4.04 All.

    Labels are indices into the full known set Y_s. Old/New for a given
    episode is then decided by that episode's own Y_s^eg -- see the note in
    the module docs of training/global_loop.py.

    Returns:
        dataset: ConcatDataset over the validation domains.
        labels: (N,) label for every sample, in dataset order.
    """
    known, _ = known_novel_split(cfg.dataset)
    cache = synthetic_root(cfg)

    parts, label_chunks = [], []
    for domain in cfg.synthetic.valid_domains:
        if not (cache / domain).is_dir():
            raise FileNotFoundError(
                f"validation domain not generated: {cache / domain}\n"
                f"run scripts/generate_synthetic_domains.py first"
            )
        ds = PACSDataset(cache, domain, known, transform=transform)
        parts.append(ds)
        label_chunks.append(ds.labels_tensor())

    return ConcatDataset(parts), torch.cat(label_chunks)


def episode_class_mapping(episode: Episode, full_known: Sequence[str]) -> torch.Tensor:
    """Map full-Y_s label indices onto this episode's label space.

    Returns:
        (|Y_s|,) tensor where entry i is the episode label for full-set class
        i, or -1 if that class is novel for this episode. Used to score
        D_valid predictions against an episode-specific classifier, and to
        split validation accuracy into Old and New.
    """
    mapping = torch.full((len(full_known),), -1, dtype=torch.long)
    for episode_label, name in enumerate(episode.known_classes):
        mapping[list(full_known).index(name)] = episode_label
    return mapping