"""Estimate the number of clusters K"""

import numpy as np
from scipy.optimize import minimize_scalar

from .clustering import kmeans_predict
from .hungarian import cluster_accuracy

def labelled_accuracy_at_k(
    features: np.ndarray,
    labelled_labels: np.ndarray,
    k: int, 
    seed: int = 0,
    n_init: int = 3,
)-> float:
    """Cluster All features into k groups, score only the labelled prefix.
    
    Args:
        features: (N, D) the first len(labelled_labels) rows are labelled.
        labelled_labels: (n_lab,) ground-truth labels for that prefix.
    """
    pred = kmeans_predict(features, k, seed=seed, n_init=n_init)
    n_lab = len(labelled_labels)
    all_acc, _, _ = cluster_accuracy(labelled_labels, pred[:n_lab], np.ones(n_lab, dtype=bool))
    return all_acc

def estimate_k(
    labelled_features: np.ndarray,
    labelled_labels: np.ndarray,
    unlabelled_features: np.ndarray,
    k_min: int,
    k_max: int = 1000,
    seed: int = 0,
    n_init: int = 3,
    verbose: bool = True,
) -> tuple[int, dict[int, float]]:
    """Brent search for the K that maximises labelled clustering accuracy.

    Args:
        labelled_features: (n_lab, D) source embeddings, known classes.
        labelled_labels: (n_lab,) their labels.
        unlabelled_features: (n_unl, D) target embeddings.
        k_min: lower bound -- |Y_s| per the paper.
        k_max: upper bound -- 1000 per the paper; clipped to the sample count.

    Returns:
        best_k: the K with the highest labelled accuracy among those evaluated.
        history: {K: labelled accuracy} for every K evaluated, sorted by K.
    """
    if len(labelled_features) != len(labelled_labels):
        raise ValueError("labelled_features and labelled_labels differ in length")
    if labelled_features.shape[1] != unlabelled_features.shape[1]:
        raise ValueError("labelled and unlabelled features have different dims")

    # Labelled rows FIRST: labelled_accuracy_at_k relies on that ordering.
    features = np.concatenate([labelled_features, unlabelled_features], axis=0)
    k_max = min(k_max, len(features))
    if not 1 <= k_min < k_max:
        raise ValueError(f"need 1 <= k_min < k_max, got {k_min}, {k_max}")

    history: dict[int, float] = {}

    def objective(x: float) -> float:
        # Brent probes real numbers; K-means needs an integer. GCD truncates.
        k = int(np.clip(int(x), k_min, k_max))
        # Nearby probes often truncate to the same K -- don't re-run K-means.
        if k not in history:
            history[k] = labelled_accuracy_at_k(
                features, labelled_labels, k, seed=seed, n_init=n_init
            )
            if verbose:
                print(f"  K={k:>4}  labelled acc = {history[k]:.4f}")
        return -history[k]      # minimize_scalar minimises

    # xatol=0.5: once the bracket is narrower than half an integer, further
    # probes can only truncate to K values already in the cache.
    minimize_scalar(
        objective, bounds=(k_min, k_max), method="bounded",
        options={"xatol": 0.5},
    )

    # Best evaluated K. Ties go to the LARGER K -- see the section notes.
    best_k = max(history, key=lambda k: (history[k], k))
    return best_k, dict(sorted(history.items()))