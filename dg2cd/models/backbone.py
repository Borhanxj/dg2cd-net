"""ViT-B/16 encoder with DINO initialisation"""

from collections.abc import Iterator, Sequence
import timm
import torch
import torch.nn as nn

class DINOViTBackbone(nn.Module):
    """ViT-B/16 encoder with DINO initialisation"""

    def __init__(
    self,
    arch: str = "vit_base_patch16_224.dino",
    pretrained: bool = True,
    trainable_blocks: Sequence[int] = (-1,),
    ):
        super().__init__()

        self.model = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
        )
        self.arch = arch
        self.embed_dim: int = self.model.embed_dim
        self.trainable_block_indices = self._resolve_block_indices(trainable_blocks)
        self._apply_freezing()

    def _resolve_block_indices(self, blocks: Sequence[int]) -> list[int]:
        n = len(self.model.blocks)
        resolved = []
        for i in blocks:
            j = i if i>= 0 else n + i
            if not 0 <= j < n:
                raise ValueError(f"Invalid block index {i} for model with {n} blocks")
            resolved.append(j)
        return sorted(set(resolved))

    def _apply_freezing(self)-> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for j in self.trainable_block_indices:
            for p in self.model.blocks[j].parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.
        Args:
            x: (B, 3, 224, 224) tensor of images
        Returns:
            (B, embed_dim) of the CLS token embeddings
        """
        tokens = self.model.forward_features(x)
        return tokens[:, 0]  # CLS token

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters to hand to the optimizer."""
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_counts(self) -> tuple[int, int]:
        """(trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def data_config(self) -> dict:
        """Input size, mean and srd the pretrained model expects."""
        return timm.data.resolve_model_data_config(self.model)

    @classmethod
    def from_config(cls, cfg, pretrained: bool = True) -> "DINOViTBackbone":
        """Build from the backbone config in the model config."""
        if cfg.feature != "cls":
            raise ValueError(f"Only CLS features are supported, got {cfg.feature}")
        return cls(
            arch=cfg.arch,
            pretrained=pretrained,
            trainable_blocks=cfg.trainable_blocks,
        )