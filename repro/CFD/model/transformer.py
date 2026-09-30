import torch
import torch.nn as nn

from .transformer_impl import Transformer as TransformerImpl


class Transformer(nn.Module):
    def __init__(
        self,
        space_dim: int = 6,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.2,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        slice_num: int = 32,
    ):
        super().__init__()

        self.model = TransformerImpl(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
        )
        self.model.apply(self.model._init_weights)

    def forward(self, inputs: dict[str, torch.Tensor]):
        centroid = inputs["centroid"]  # (B, N, 3)
        normal = inputs["normal"]  # (B, N, 3)

        out_pressure, out_wss = self.model(centroid, normal)  # (B, N, 1), (B, N, 3)

        outputs = {
            "pressure": out_pressure,  # (B, N, 1)
            "wss": out_wss,  # (B, N, 3)
        }

        return outputs
