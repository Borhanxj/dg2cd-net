"""
Gradient Reversal Layer

Classifier F_c: minimizes L_s + L_adv
Encoder F: minimizes L_s - L_adv
GRL is enserted in between F and F_c.
"""
import torch

class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor,  lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x) # Identity

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None 


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward, gradient negated by lambd in backward.
    Args:
        x: features to pass to the adversarial classifier head
        lambd: strength of the gradient reversal
    Returns:
        output tensor
    """
    return _GradientReversal.apply(x, lambd)