"""Seeding and device resolution"""

import random 
import numpy as np
import torch

def set_seed(seed:int, deterministic: bool = False):
    """Seed python, numpy and torch
    
    Args:
        seed (int): seed value
        deterministic (bool): whether to set deterministic mode for torch
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "cuda") -> torch.device:
    """
    Resolve the appropriate device based on the requested device and availability.

    Args:
        requested (str): The requested device.

    Returns:
        torch.device: The resolved device.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)