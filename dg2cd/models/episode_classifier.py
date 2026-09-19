"""Episode-specific open-set classifier """

import torch
import torch.nn as nn
from .grl import grad_reverse

class EpisodeClassifier(nn.Module):
    def __init__(
        self, 
        in_features: int,
        num_known_classes: int, 
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_known_classes < 1:
            raise ValueError("num_known_classes must be >= 1")

        self.in_features = in_features
        self.num_known_classes = num_known_classes
        self.num_outputs = num_known_classes + 1
        self.unknown_index = num_known_classes # Last index

        if hidden_dim is None:
            self.net = nn.Module = nn.Linear(in_features, self.num_outputs)
        else: 
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_outputs),
            )

    def forward(
        self, 
        features: torch.Tensor,
        reverse_gradient: bool = False,
        grl_lambda: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the classifier.
        Args:
            features: (B, in_features) tensor of input features
            reverse_gradient: If True, apply gradient reversal to the input features
            grl_lambda: Lambda for gradient reversal layer
        Returns:
            (B, num_outputs) tensor of logits
        """
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError(
                f"expected (B, {self.in_features}) features, "
                f"got {tuple(features.shape)}"
            )
        
        if reverse_gradient:
            features = grad_reverse(features, grl_lambda)

        return self.net(features) # Forward pass through the classifier

    def extra_repr(self) -> str: 
        return (
            f"in_features={self.in_features}, "
            f"known={self.num_known_classes}, outputs={self.num_outputs}"
        )