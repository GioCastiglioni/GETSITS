# Adapted from https://github.com/HMS97/ConvNeXtV2-Unet

from typing import Sequence
from logging import Logger

import torch
from torch import nn

from getsits.encoders.base import Encoder
from getsits.encoders.ltae import LTAE2d

from torch.nn import functional as F
from timm.layers import DropPath


class ConvNext(Encoder):
    """
    Multi Temporal Convnext Encoder for Supervised Baseline, to be trained from scratch.
    It supports single time frame inputs with optical bands

    Args:
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        input_size (int): Size of the input images (height and width).
        topology (Sequence[int]): The number of feature channels at each stage of the U-Net encoder.

    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        multi_temporal: int,
        topology: Sequence[int],
        output_dim: int | list[int],
        download_url: str,
        encoder_weights: str | None = None,
        projection_dim: int = 64,
        depths: list = [2, 2, 8, 2],
        positional_encoding: str | None = "normal"
    ):
        super().__init__(
            model_name="ConvNeXtV2",
            encoder_weights=encoder_weights,  # no pre-trained weights, train from scratch
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=output_dim,
            output_layers=topology,
            multi_temporal=multi_temporal,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
            positional_encoding=positional_encoding
        )
        self.depths = depths
        self.num_stage = len(self.depths)
        self.in_channels = len(input_bands["optical"])
        self.topology = topology
        drop_path_rate = 0.

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.topology[0], kernel_size=2, stride=2),
            LayerNorm(self.topology[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.topology[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.topology[i], self.topology[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[Block(dim=self.topology[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.norm = nn.LayerNorm(self.topology[-1], eps=1e-6) # final norm layer

        self.tmap = LTAE2d(
            in_channels=self.topology[-1],
            d_model=256,
            n_head=16,
            mlp=[256, self.topology[-1]],
            return_att=True,
            d_k=4,
            positional_encoding=positional_encoding,
            layer_norm=True
        )
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.topology[-1], 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, projection_dim)
        )
        self.projector.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, input, batch_positions=None):
        input = input.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = input.shape

        pad_mask = (
            (input == 0).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # (B, T) pad mask

        x = input.reshape(B * T, C, H, W)

        feature_maps = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_maps.append(x)
        
        feature_maps = [
            fm.reshape(B, T, -1, fm.shape[-2], fm.shape[-1]) for fm in feature_maps
        ]

        out, att = self.tmap(
            feature_maps[-1].permute(0, 2, 1, 3, 4),        # (B, C, T, H, W)
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )

        return out, feature_maps, pad_mask, att

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = True) -> None:
        if not from_scratch:
            logger.info(f"Loading pre-trained weights from {self.encoder_weights}...")
            model_dict = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)["model"]
            model_dict = {k[8:]: v for k,v in model_dict.items() if k.startswith("encoder.")}
            self.load_state_dict(model_dict)
            logger.info("Pre-trained weights loaded successfully.")
        else:pass


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) # pointwise/1x1 convs
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = input + self.drop_path(x)
        return x


