"""
Unsupervised contrastive loss
"""

import torch
import torch.nn.functional as F

def unsupervised_contrastive_loss(
    features_view1: torch.Tensor,
    features_view2: torch.Tensor,
    temperature: float = 0.07,
    include_positive_in_denominator: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Unsupervised contrastive loss (InfoNCE) for two views of the same batch.

    Args:
        features_view1: (B, D) embeddings for view 1
        features_view2: (B, D) embeddings for view 2
        temperature: temperature for the softmax
        include_positive_in_denominator: whether to include the positive sample in the denominator
        reduction: reduction method for the loss

    Returns:
        Scalar, or (2*B,) when reduction == "none".
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if features_view1.shape != features_view2.shape:
        raise ValueError("features_view1 and features_view2 must have the same shape")
    if features_view1.ndim != 2:
        raise ValueError(f"features must be (B, D), got {tuple(features_view1.shape)}")

    batch_size = features_view1.shape[0]
    if batch_size < 2 and not include_positive_in_denominator:
        raise ValueError("batch size must be >= 2 when not including positive in denominator")

    # Concatenate the two views and normalize
    z = torch.cat([features_view1, features_view2], dim=0)
    z = F.normalize(z, dim=-1)

    # Compute the similarity matrix
    n = 2 * batch_size
    sim = (z @ z.t()) / temperature

    # Compute the positive logits
    idx = torch.arange(n, device=z.device)
    pos_idx = (idx + batch_size) % n
    pos_logits = sim[idx, pos_idx]

    # Exclude the positive samples from the denominator
    excluded = torch.zeros(n, n, dtype=torch.bool, device=z.device)
    excluded[idx, pos_idx] = True

    denominator = torch.logsumexp(sim.masked_fill(excluded, float("-inf")), dim=1)

    per_sample_loss = -pos_logits + denominator

    if reduction == "mean":
        return per_sample_loss.mean()
    if reduction == "sum":
        return per_sample_loss.sum()
    if reduction == "none":
        return per_sample_loss
    raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")