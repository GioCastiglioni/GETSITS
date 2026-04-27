import torch
import torch.nn as nn
from torch.nn import functional as F

from getsits.encoders.base import Encoder


class Decoder(nn.Module):
    """Base class for decoders."""

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ) -> None:
        """Initialize the decoder.

        Args:
            encoder (Encoder): encoder used.
            num_classes (int): number of classes of the task.
            finetune (bool): whether the encoder is finetuned.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.finetune = finetune

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, num_layers=2, proj_channels=256):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, proj_channels, kernel_size=1))
            layers.append(nn.GroupNorm(proj_channels//16, proj_channels))
            layers.append(nn.ReLU(inplace=False))
            in_channels = proj_channels
        layers.append(nn.Conv2d(in_channels, proj_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.net(x)
        out = F.normalize(out, dim=1)  # L2 normalize across channels
        return out
    
class BCLProj(nn.Module):
    def __init__(self, in_channels=64, hidden_d=512, out_d=128):
        super().__init__()
        self.in_layer = nn.Conv2d(in_channels, hidden_d, kernel_size=1)
        self.norm = nn.GroupNorm(hidden_d//16, hidden_d)
        self.ReLU = nn.ReLU(inplace=False)
        self.out = nn.Conv2d(hidden_d, out_d, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.in_layer(x)
        out = self.norm(out)
        out = self.ReLU(out)
        out = self.out(out)
        out = F.normalize(out, dim=1)  # L2 normalize across channels
        return out
