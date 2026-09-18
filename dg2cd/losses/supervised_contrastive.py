"""
Supervised contrastive loss function.
"""

import torch 
import torch.nn.functional as F

# This function computes the class prototypes for a given set of features and labels.
def class_prototypes(
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: # Returns a tuple containing prototypes and counts
    """Mean embedding per class

    Args:
        features: (B, D) embeddings
        labels: (B,) int64 in [0, num_classes)
        num_classes: |Y_s^eg|
        normalize: average unit-normalised embeddings rather than raw ones
        counts: (num_classes,) number of samples per class

    Returns:
        prototypes: (C, D); rows of classes absent from the batch are zeroed
        counts: (C,) bool, True for classes that do occur in the batch
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    # shape: (num_classes, embedding_dim) 
    sums = features.new_zeros(num_classes, features.shape[-1])
    sums.index_add_(0, labels, features)

    # shape: (num_classes,sample_count_per_class)
    counts = features.new_zeros(num_classes)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))

    present = counts > 0 
    prototypes = sums / counts.clamp(min=1).unsqueeze(-1)

    return prototypes, present


# This function computes the supervised prototype contrastive loss as described in DG2CD-Net Eq. 4.
def supervised_prototype_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int | None = None,
    prototypes: torch.Tensor | None = None,
    temperature: float = 0.07,
    detach_prototypes: bool = False,
    normalize_before_mean: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """DG2CD-Net Eq. 4.

    Args:
        features: (B, D) embeddings
        labels: (B,) int64 in [0, num_classes)
        num_classes: |Y_s^eg|
        prototypes: optional (C, D) externally maintained mu_y
        temperature: tau
        detach_prototypes: stop gradients through mu_y
        reduction: "mean" | "sum" | "none"

    Returns:
        Scalar, or (B,) when reduction == "none".
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if features.ndim != 2:
        raise ValueError(f"features must be (B, D), got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be (B,) and match the features batch dim")
    if labels.numel() == 0:
        raise ValueError("empty batch")

    labels = labels.long()

    if prototypes is not None:
        if prototypes.ndim != 2 or prototypes.shape[-1] != features.shape[-1]:
            raise ValueError("prototypes must be (C, D) matching the feature dim")
        num_classes = prototypes.shape[0]
    elif num_classes is None:
        num_classes = int(labels.max().item()) + 1

    if prototypes is None:
        prototypes, present = class_prototypes(
            features, labels, num_classes, normalize_before_mean
        )
        if detach_prototypes:
            prototypes = prototypes.detach()
    else:
        present = torch.ones(num_classes, dtype=torch.bool, device=features.device)

    z = F.normalize(features, dim=-1)
    mu = F.normalize(prototypes, dim=-1)

    logits = (z @ mu.t()) / temperature
    logits = logits.masked_fill(~present.unsqueeze(0), float("-inf"))

    return F.cross_entropy(logits, labels, reduction=reduction)    