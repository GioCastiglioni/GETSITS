# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from logging import Logger
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from timm.layers import drop_path, to_2tuple, trunc_normal_
from timm.layers import trunc_normal_ as __call_trunc_normal_

from .base import Encoder
from getsits.encoders.ltae import LTAE2d


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class VIT_Encoder(Encoder):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        encoder_weights,
        model_name,
        input_size,
        input_bands,
        embed_dim,
        output_layers,
        output_dim,
        download_url,
        patch_size=16,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
        projection_dim: int = 64,
        positional_encoding: str | None = "normal",
    ):
        Encoder.__init__(
            self,
            model_name=model_name,
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.in_chans = len(input_bands["optical"])
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_chans=self.in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.topology = [output_dim for _ in self.output_layers]
        self.init_std = init_std

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
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

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, batch_positions):
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        pad_mask = (
            (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # (B, T) pad mask

        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.output_layers:
                out = x[:, 1:]
                out = (
                    out.transpose(1, 2)
                    .view(
                        x.shape[0],
                        -1,
                        self.input_size // self.patch_size,
                        self.input_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)
        output = [
            fm.reshape(B, T, -1, fm.shape[-2], fm.shape[-1]) for fm in output
        ]

        out, att = self.tmap(
            output[-1].permute(0, 2, 1, 3, 4),        # (B, C, T, H, W)
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )

        return out, output, pad_mask, att
    
    def load_encoder_weights(self, logger: Logger, from_scratch: bool = True) -> None:
        if not from_scratch:
            logger.info(f"Loading pre-trained weights from {self.encoder_weights}...")
            model_dict = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)["model"]
            model_dict = {k[8:]: v for k,v in model_dict.items() if k.startswith("encoder.")}
            self.load_state_dict(model_dict)
            logger.info("Pre-trained weights loaded successfully.")
        else:pass

