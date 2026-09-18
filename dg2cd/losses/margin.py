""" 
Confidence margin loss
"""
import torch

def confidence_margin_loss(
    logits: torch.Tensor,
    margin: float = 0.7,
    reduction: str = "mean",
)-> torch.Tensor:
    """Confidence margin loss for the adversarial classifier head.

    Args:
        logits: (B, K+1) logits for the adversarial classifier head, where K is the number of known classes and the last column corresponds to the unknown class.
        margin: confidence margin for the known classes
        reduction: reduction method for the loss
        
    Returns:
        scalar loss value
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be (B, K+1), got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("need at least one known class plus the unknown class")
    if not 0.0 < margin < 1.0:
        raise ValueError(f"margin must be in (0, 1), got {margin}")

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    p_unknown = probs[:, -1]
    p_max_known = probs[:, :-1].max(dim=1).values
    gap = (p_max_known - p_unknown).abs()

    # Compute the per-sample loss based on the margin
    per_sample = torch.clamp(margin - gap, min=0.0)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction: {reduction}")