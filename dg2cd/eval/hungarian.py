"""Clustering accuracy with Hungarian matching.
One optimal cluster to class assignment is computed over the whole set, and 
Old and New accuracies are read off that same assignment, restricted to samples
Whose TRUE class is Old or New, respectively.
"""

from collections.abc import Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        old_mask: np.ndarray,
) -> tuple[float, float, float]:
    """
    All / Old / New accuracy under a single Hungarian assignment.

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) cluster ids form K-means.
        old_mask: (N,) boolean, True where the sample's true class is Old

    Returns:
        (all_acc, old_acc, new_acc) as fractions in [0, 1].
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    old_mask = np.asarray(old_mask, dtype=bool)

    if not (y_true.shape == y_pred.shape == old_mask.shape):
        raise ValueError("y_true, y_pred, and old_mask must have the same shape.",)
    if y_true.size == 0:
        raise ValueError("y_true, y_pred, and old_mask must be non-empty.",)

    # Square contingency matrix: w[c, k] = number of samples with true class c and predicted cluster k
    # Square so the assignment is a full permuation even when K != #classes
    d = int(max(y_pred.max(), y_true.max()) + 1) # number of classes/clusters
    w = np.zeros((d, d), dtype=np.int64) # contingency matrix
    np.add.at(w, (y_pred, y_true), 1)  # increment counts for each (true class, predicted cluster) pair

    rows, cols = linear_sum_assignment(w, maximize=True) 
    cluster_to_class = np.full(d, -1, dtype=np.int64) 
    cluster_to_class[rows] = cols

    correct = cluster_to_class[y_pred] == y_true

    all_acc = float(correct.mean())
    old_acc = float(correct[old_mask].mean()) if old_mask.any() else float("nan")
    new_acc = float(correct[~old_mask].mean()) if (~old_mask).any() else float("nan")
    return all_acc, old_acc, new_acc


def old_class_indices(dataset_classes: Sequence[str], old_classes: Sequence[str]) -> list[int]:
    """
    Returns the indices of samples whose true class is Old. 
    
    Example: 
        dataset_classes = ["cat", "dog", "fish"]
        old_classes = ["cat", "fish"] -> returns [0, 2]
    """
    missing = set(old_classes) - set(dataset_classes)
    if missing:
        raise ValueError(f"Old classes {missing} not found in dataset classes {dataset_classes}.")
    return [list(dataset_classes).index(c) for c in old_classes]