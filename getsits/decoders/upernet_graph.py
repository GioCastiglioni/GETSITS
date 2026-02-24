# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from getsits.decoders.base import Decoder
from getsits.encoders.base import Encoder

def batched_index_select(values, indices):
    """
    Selects feature vectors based on KNN indices considering the batch dimension.
    Args:
        values: (B, N, C)
        indices: (B, N, k)
    Returns:
        (B, N, k, C)
    """
    B, N, C = values.shape
    k = indices.shape[2]
    
    values_flat = values.reshape(B * N, C)
    
    # offset: (B, 1, 1) -> broadcastable to (B, N, k)
    batch_offset = torch.arange(B, device=values.device).unsqueeze(1).unsqueeze(2) * N
    
    # Add offset to indices and flatten: (B*N*k)
    indices_flat = (indices + batch_offset).reshape(-1)
    
    selected = torch.index_select(values_flat, 0, indices_flat)
    
    return selected.reshape(B, N, k, C)


class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, k_semantic=8, reduction_ratio=4):
        super().__init__()
        self.k_semantic = k_semantic
        self.total_k = 8 + k_semantic 
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.reduction_dim = max(16, in_channels // reduction_ratio)
        self.theta = nn.Conv2d(in_channels, self.reduction_dim, 1) 
        
        self.phi = nn.Conv2d(in_channels, out_channels, 1)
        
        self.att_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1)
        )
        
        self.out_proj = nn.Conv2d(out_channels, out_channels, 1)
        self.norm = nn.SyncBatchNorm(out_channels)
        self.act = nn.ReLU()

    def get_hybrid_indices(self, x_flat, H, W):
        # x_flat: (B, C_red, N)
        B, _, N = x_flat.shape
        device = x_flat.device

        # 1. Spatial neighbors via explicit unfolding
        # (1, 1, H, W)
        idx_img = torch.arange(N, device=device, dtype=torch.float32).view(1, 1, H, W)
        
        idx_padded = F.pad(idx_img, (1, 1, 1, 1), mode='replicate')
        
        # (1, 9, N)
        patches = F.unfold(idx_padded, kernel_size=3)
        # (1, N, 9)
        patches = patches.transpose(1, 2).long()
        
        # Center pixel is strictly at index 4 in a flattened 3x3 patch.
        # spatial_idx: (1, N, 8) -> (B, N, 8)
        spatial_idx = torch.cat([patches[:, :, :4], patches[:, :, 5:]], dim=2)
        spatial_idx = spatial_idx.expand(B, -1, -1)

        # 2. Semantic neighbors via Cosine Similarity
        x_norm = F.normalize(x_flat, p=2, dim=1)
        # sim: (B, N, N)
        sim = torch.bmm(x_norm.transpose(1, 2), x_norm)

        # 3. Mask spatial neighbors and self to avoid semantic duplication
        # mask: (B, N, N)
        mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        mask.scatter_(2, spatial_idx, True)
        mask.diagonal(dim1=1, dim2=2).fill_(True)
        
        sim.masked_fill_(mask, float('-inf'))

        # 4. Top-K Semantic
        # semantic_idx: (B, N, k_semantic)
        _, semantic_idx = sim.topk(self.k_semantic, dim=-1, largest=True)

        # final_idx: (B, N, 8 + k_semantic)
        final_idx = torch.cat([spatial_idx, semantic_idx], dim=-1)

        return final_idx

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        
        # (B, C_red, N)
        x_reduced = self.theta(x).view(B, -1, N) 
        
        # hybrid_idx: (B, N, total_k)
        hybrid_idx = self.get_hybrid_indices(x_reduced, H, W)
        
        # (B, C_out, N)
        x_phi = self.phi(x).view(B, -1, N) 
        C_out = x_phi.shape[1]
        
        # (B, N, C_out)
        x_phi_t = x_phi.permute(0, 2, 1) 
        
        # (B, N, total_k, C_out)
        neighbor_feats = batched_index_select(x_phi_t, hybrid_idx) 
        
        # (B, N, 1, C_out)
        center_feats = x_phi_t.unsqueeze(2)
        
        # (B, N, total_k, 2 * C_out)
        edge_feats = torch.cat([center_feats.expand(-1, -1, self.total_k, -1), neighbor_feats], dim=-1)
        
        # (B, N, total_k)
        att_weights = self.att_mlp(edge_feats).squeeze(-1) 
        att_weights = F.softmax(att_weights, dim=-1)
        
        # (B, N, C_out)
        agg_feats = (neighbor_feats * att_weights.unsqueeze(-1)).sum(dim=2) 
        
        # (B, C_out, H, W)
        agg_feats = agg_feats.permute(0, 2, 1).view(B, C_out, H, W)
        
        out = self.out_proj(agg_feats)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class SegUPerNetGraph(Decoder):
    """Unified Perceptual Parsing for Scene Understanding with optional lateral GNN."""

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: list[int] | None = None,
        use_gnn_lateral: bool = True,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "UPerNet_GNN"
        self.encoder = encoder
        self.no_tmap = (str(encoder) == "OlmoEarth") or (str(encoder) == "GalileoTiny") or (str(encoder) == "AnySat_Encoder")
        self.finetune = finetune
        self.feature_multiplier = feature_multiplier

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)
        self.topology = self.encoder.topology

        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if self.encoder.pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.align_corners = False
        self.channels = channels
        self.dec_topology = [self.channels]
        self.num_classes = num_classes

        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        num_laterals = len(self.in_channels[:-1])
        
        for i, in_channels in enumerate(self.in_channels[:-1]): 
            
            is_deepest_lateral = (i == num_laterals - 1)
            
            if use_gnn_lateral and is_deepest_lateral:
                l_conv = GraphAttention(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    k_semantic=8 # 8 neighbors + self
                )
            else:
                l_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=False),
                )
            
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def _forward_feature(self, inputs):
        """Forward function combining laterals and PSP."""
        
        # Build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        # PSP path (deepest level)
        laterals.append(self.psp_forward(inputs))

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # Build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # Append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward_fmaps(self, img: dict[str, torch.Tensor]) -> torch.Tensor:

        if type(img) is dict: pass
        else: img = {'optical': img.requires_grad_(True)}

        # img[modality] of shape [B C T=1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            # Remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        feat = self.neck(feat)
        feat = self._forward_feature(feat)

        return feat
    
    def forward_features(self, x, batch_positions=None):

        if type(x) is dict: pass
        else: x = {'optical': x}

        feat = self.forward_fmaps(x)
        
        output_shape = x[list(x.keys())[0]].shape[-2:]

        feat = F.interpolate(feat, size=output_shape, mode="bilinear", align_corners=False)
        
        return feat

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None, batch_positions=None, return_feats=False
    ) -> torch.Tensor:
        """
        Compute the segmentation output.
        Args:
            img: {modality: (B, C, T=1, H, W)}
            output_shape: Target spatial dims (H, W).
        Returns:
            (B, num_classes, H', W')
        """
        
        feat = self.forward_features(img)

        output = self.conv_seg(feat)

        return output


class SegMTUPerNetGraph(SegUPerNetGraph):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        multi_temporal: int,
        multi_temporal_strategy: str | None,
        segmentation: bool = True,
        pool_scales: list[int] = [1, 2, 3, 6],
        feature_multiplier: int = 1,
        use_gnn_lateral: bool = True
    ) -> None:
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            channels=channels,
            pool_scales=pool_scales,
            feature_multiplier=feature_multiplier,
            in_channels=encoder.topology,
            use_gnn_lateral=use_gnn_lateral
        )
        self.segmentation = segmentation
        if not self.segmentation:
            self.encoder.projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(self.topology[-1], 512),
                nn.LayerNorm(normalized_shape=512),
                nn.GELU(),
                nn.Dropout(p=0.15),
                nn.Linear(512, 512),
                nn.LayerNorm(normalized_shape=512),
                nn.GELU(),
                nn.Dropout(p=0.15),
                nn.Linear(512, num_classes)
            ).requires_grad_(True)

        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy

    def forward_features(
        self, img: dict[str, torch.Tensor], batch_positions=None, output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """
        Compute the segmentation output for multi-temporal data.
        img: {modality: (B, C, T, H, W)}
        """

        # If the encoder handles multi_temporal we feed it with the input
        if not self.finetune:
            with torch.no_grad():
                _, feat, _, att = self.encoder(img, batch_positions)
                if not self.no_tmap: feat = self.collapse_T(feat, att)
        else:
            _, feat, _, att = self.encoder(img, batch_positions)
            if not self.no_tmap: feat = self.collapse_T(feat, att)

        feat = self.neck(feat)
        feat = self._forward_feature(feat)

        if output_shape is None:
            output_shape = img.shape[-2:]

        # Interpolate to the target spatial dims
        feat = F.interpolate(feat, size=output_shape, mode="bilinear")

        return feat


    def forward(
        self, img: dict[str, torch.Tensor], batch_positions=None, output_shape: torch.Size | None = None, return_feats=False
    ) -> torch.Tensor:

        if self.segmentation:
            feat = self.forward_features(img, batch_positions, output_shape)
            output = self.conv_seg(feat)
        else:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img, batch_positions)[0]
            else:
                feat = self.encoder(img, batch_positions)[0]
            
            output = self.encoder.projector(feat)

        return output

    def collapse_T(self, feature_maps, att):
        n_heads = att.shape[0]
        collapsed_maps = []
        
        for fm in feature_maps:
            # fm: (B, T, C, H, W)
            B, T, C, H_fm, W_fm = fm.shape
            H_att, W_att = att.shape[-2], att.shape[-1]
            
            if (H_fm, W_fm) != (H_att, W_att):
                att_resized = att.view(-1, 1, H_att, W_att)
                att_resized = F.interpolate(att_resized, size=(H_fm, W_fm), mode='bilinear', align_corners=False)
                att_spatial = att_resized.view(n_heads, B, T, 1, H_fm, W_fm)
            else:
                att_spatial = att.unsqueeze(3) # (Heads, B, T, 1, H, W)

            if C % n_heads == 0:
                c_per_head = C // n_heads
                
                fm_split = fm.view(B, T, n_heads, c_per_head, H_fm, W_fm)
                fm_split = fm_split.permute(2, 0, 1, 3, 4, 5)
                
                weighted = fm_split * att_spatial 
                
                collapsed = weighted.sum(dim=2)
                collapsed_fm = collapsed.permute(1, 0, 2, 3, 4).reshape(B, C, H_fm, W_fm)
                
            else:
                att_avg = att_spatial.mean(dim=0) # (B, T, 1, H, W)
                collapsed_fm = (fm * att_avg).sum(dim=1)

            collapsed_maps.append(collapsed_fm)
            
        return collapsed_maps
    

class PPM(nn.ModuleList):
    """
    Pooling Pyramid Module.
    """
    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class Feature2Pyramid(nn.Module):
    """
    Neck structure to connect backbone and decoder.
    """
    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        nn.SyncBatchNorm(embed_dim[i]),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return outputs