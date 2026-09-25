"""Task vectors and validation-weighted merging

    delta_e         = theta_global - theta_local_e
    w_e             = softmax(All_e)
    theta_global    = theta_global - sum_e(w_e * delta_e)

    """

from collections.abc import Sequence 
import torch
import torch.nn as nn

StateDict = dict[str, torch.Tensor]

# This function takes a model and returns the names of 
# all parameters that require gradients. 
def trainable_names(model: nn.Module) -> list[str]:
    """Names of parameters that recieve gradients."""
    out = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            out.append(name)
    return out

@torch.no_grad()
def snapshot(model: nn.Module, names: Sequence[str])-> StateDict:
    """Copy the neamed parameters to CPU. 
    
    Returns independent tensors. Later training of model will not affect the snapshot.
    """
    params = dict(model.named_parameters())
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    return {n: params[n].detach().to("cpu", copy=True) for n in names}


@torch.no_grad()
def load_params(model: nn.Module, state: StateDict) -> None:
    """Write 'state' into the model's parameters in place."""
    params = dict(model.named_parameters())
    for name, value in state.items():
        if name not in params:
            raise ValueError(f"Missing parameter: {name}")
        target = params[name]
        if target.shape != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {target.shape} vs {value.shape}")
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def task_vector(theta_global: StateDict, theta_local: StateDict) -> StateDict:
    """delta = theta_global - theta_local"""
    if theta_global.keys() != theta_local.keys():
        raise KeyError("theta_global and theta_local track different parameters")
    return {n: theta_global[n] - theta_local[n] for n in theta_global}


def softmax_weights(all_scores: Sequence[float]) -> torch.Tensor:
    """w_e = softmax(All_e)
    
    Args:
        all_scores: one All accuracy per episode as a fraction in [0, 1]
        
    Returns:
        (ne,) float64 weights that sum to 1.0"""
    scores = torch.tensor(list(all_scores), dtype=torch.float64)
    if scores.numel() == 0:
        raise ValueError("No scores provided")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be in [0, 1]")

    return torch.softmax(scores, dim=0)


def state_norm(state: StateDict) -> float:
    """Compute the L2 norm of a state dict."""
    return float(torch.sqrt(sum((v.double() ** 2).sum() for v in state.values())))

def merge(
        theta_global: StateDict,
        deltas: Sequence[StateDict],
        weights: torch.Tensor
) -> StateDict:
    """theta_global = theta_global - sum_e(w_e * delta_e)
    
    Args:
        theta_global: the global model parameters
        deltas: a list of task vectors (theta_global - theta_local_e)
        weights: a tensor of weights for each task vector

    Returns:
        theta_global after merging the weighted task vectors
    """
    if len(deltas) != len(weights):
        raise ValueError("Length of deltas and weights must match")

    merged = {n: v.clone() for n, v in theta_global.items()}
    for w, delta in zip(weights.tolist(), deltas):
        if delta.keys() != merged.keys():
            raise KeyError("Delta and theta_global track different parameters")
        for n in merged:
            merged[n] -= w * delta[n]
    return merged