"""
Adversarial loss
"""
import torch

def open_set_adversarial_loss(
    logits: torch.Tensor,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Open-set adversarial loss for the adversarial classifier head.

    Args:
        logits: (B, K+1) logits for the adversarial classifier head, where K is the number of known classes and the last column corresponds to the unknown class.
        alpha: weight for the open-set loss
        reduction: reduction method for the loss
    Returns:
        Scalar, or (B,) when reduction == "none".
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (B, K+1), got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("need at least one known class plus the unknown class")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # log p(unknown) -- the last column of the log-softmax.
    log_p_unknown = torch.log_softmax(logits, dim=1)[:, -1]

    # log(1 - p(unknown)) computed exactly, without ever forming (1 - p).
    # 1 - p(unknown) is the total probability of the known classes, so
    #   log(1 - p_unk) = logsumexp(known logits) - logsumexp(all logits)
    log_p_known = torch.logsumexp(logits[:, :-1], dim=1) - torch.logsumexp(logits, dim=1)

    per_sample = -(alpha * log_p_unknown + (1.0 - alpha) * log_p_known)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction: {reduction}")