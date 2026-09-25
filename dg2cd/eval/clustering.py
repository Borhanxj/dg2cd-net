"""Embedding extraction and K-means clustering for GCD evaluation."""

from collections.abc import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from .hungarian import cluster_accuracy


@torch.no_grad()
def extract_embeddings(
    encoder, 
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    normalize: bool = True
):
    """Embed every image in 'dataset', in dataset order
    
    Args:
        encoder: returns (B, D) [CLS] embeddings.
        dataset: yields (image, label). Must use the EVAL transform -- a
            two-view training transform is tolerated (first view is used) but
            random augmentation at evaluation time makes results noisy.
        normalize: L2-normalise before clustering, as GCD does. K-means uses
            Euclidean distance; on unit vectors that is a monotone function
            of cosine similarity, which is what every loss here optimises.

    Returns:
        features: (N, D) float32.
        labels: (N,) int64.
    """
    was_training = encoder.training
    encoder.eval()
    encoder.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    features, labels = [], []
    for image, y in loader:
        if isinstance(image, (tuple, list)):
            image = image[0]
        z = encoder(image.to(device, non_blocking=True))
        if normalize:
            z = F.normalize(z, dim=-1)
        features.append(z.float().cpu())
        labels.append(torch.as_tensor(y))

    encoder.train(was_training)
    return torch.cat(features).numpy(), torch.cat(labels).numpy().astype(np.int64)


def kmeans_predict(
        features: np.ndarray,
        k: int, 
        seed: int = 0, 
        n_init: int = 10,
) -> np.ndarray:
    """Cluster assignments form K-means with a fixed seed."""
    if k <1:
        raise ValueError(f"Invalid number of clusters: {k}")
    if k > len(features):
        raise ValueError(f"Number of clusters {k} exceeds number of samples {len(features)}")
    km = KMeans(n_clusters=k,n_init=n_init, random_state=seed)
    return km.fit_predict(features)


def evaluate_clustering(
        features: np.ndarray,
        labels: np.ndarray,
        old_classes: Sequence[int],
        k: int,
        seed: int = 0,
        n_init: int = 10,
) -> dict[str, float]:
    """K-means + Hungarian on precomputed embeddings.
    
    Args:
        features, labels: from extract_embeddings().
        old_classes: label indices that count as "old" classes.
        k: number of clusters to use in K-means. estimated via Brent's method in GCD.
    
    Returns:
        {"all", "old", "new", "k"}; accuracies as fractions in [0, 1].
        
    """

    preds = kmeans_predict(features, k, seed=seed, n_init=n_init)
    old_mask = np.isin(labels, np.asarray(list(old_classes)))
    all_acc, old_acc, new_acc = cluster_accuracy(labels, preds, old_mask)
    return {"all": all_acc, "old": old_acc, "new": new_acc, "k": int(k)}