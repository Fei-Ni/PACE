"""Prediction heads used by the PACE progress readouts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictionHead(nn.Module):
    """
    Predict a target latent vector from a VLM token representation.

    PACE uses it for the goal queries -> foresight anchor in frozen DINOv2 CLS space. The output is L2-normalised to
    stay in the same geometry as the frozen DINOv2 targets.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.normalize_output = bool(normalize_output)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        net_param = next(self.net.parameters())
        pred = self.net(hidden.to(dtype=net_param.dtype))
        pred = pred.float()
        if self.normalize_output:
            pred = F.normalize(pred, dim=-1)
        return pred
