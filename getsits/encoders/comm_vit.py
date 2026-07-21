import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple
from einops import rearrange, repeat
from collections import OrderedDict
from torchvision.models import AlexNet
from torch.utils.checkpoint import checkpoint

from getsits.encoders.base import Encoder
from getsits.encoders.ltae import LTAE2d

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def build_2d_sincos_posemb(h, w, embed_dim, temperature=10000.):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                         torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    return pos_emb.reshape(1, h, w, embed_dim).permute(0, 3, 1, 2)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# =========================================================================
# COMM BLOCKS & ADAPTERS
# =========================================================================

class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention module between 2 inputs. """
    def __init__(self, d_model: int, n_heads: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1x = nn.LayerNorm(d_model)
        self.ln_1y = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                  attn_mask: torch.Tensor = None):
        return self.attn(x, y, y, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1x(x), self.ln_1y(y), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, d_model: int, n_head: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        return self.attn(x.clone(), x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class FusionTransformer(nn.Module):
    """Fusion of features from multiple modalities using attention."""
    def __init__(self, width: int, n_heads: int, n_layers: int, fusion: str = "concat",
                 pool: str = "cls", add_bias_kv: bool = False, dropout: float = 0.,
                 batch_first: bool = True):
        super().__init__()
        self.fusion = fusion
        self.width = width
        self.layers = n_layers
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) if self.pool == "cls" else None
        
        if fusion == "concat":
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                       dropout=dropout, batch_first=batch_first)
                for _ in range(n_layers)])
        elif fusion == "x-attn":
            self.resblocks = nn.ModuleList([
                nn.Sequential(*[
                    ResidualCrossAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                                dropout=dropout, batch_first=batch_first)
                    for _ in range(n_layers)])
                for _ in range(2)])
        else:
            raise ValueError("Unknown fusion %s" % fusion)
        self.initialize()

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        
        blocks = self.resblocks if self.fusion == "concat" else [b for seq in self.resblocks for b in seq]
        for block in blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: List[torch.Tensor], key_padding_mask: List[torch.Tensor] = None, output_layers: list = None):
        if self.fusion == "concat":
            x = torch.cat(x, dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
            if self.pool == "cls":
                cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_token, x), dim=self.token_dim)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat((torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()

            out_features = []
            for i, layer in enumerate(self.resblocks):
                # Gradient checkpointing optional
                x = layer(x, key_padding_mask=key_padding_mask)
                if output_layers is not None and i in output_layers:
                    out_features.append(self.norm(x))

            return out_features if output_layers is not None else self.norm(x)

        elif self.fusion == "x-attn":
            x1, x2 = x
            out_features = []
            for i in range(self.layers):
                x1 = self.resblocks[0][i](x1, x2, key_padding_mask)
                x2 = self.resblocks[1][i](x2, x1, key_padding_mask)
                
                if output_layers is not None and i in output_layers:
                    out_features.append(self.norm(torch.cat([x1, x2], dim=self.token_dim)))
            
            return out_features if output_layers is not None else self.norm(torch.cat([x1, x2], dim=self.token_dim))


class PatchedInputAdapter(nn.Module):
    def __init__(self, num_channels: int, stride_level: int, patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None, sincos_pos_emb: bool = True, learnable_pos_emb: bool = False,
                 image_size: Union[int, Tuple[int]] = 224):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)

        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        self.dim_tokens = dim_tokens
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.proj = nn.Conv2d(in_channels=self.num_channels, out_channels=self.dim_tokens,
                              kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        B, C, H, W = x.shape
        N_H, N_W = H // self.P_H, W // self.P_W
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')
        return x_patch + x_pos_emb


class AlexNetEncoder(AlexNet):
    def __init__(self, in_channels: int, latent_dim: int = 256, dropout: float = 0.5, global_pool: str = ""):
        super().__init__(dropout=dropout)
        self.global_pool = global_pool
        # Adapting the first layer to accommodate dynamic input channels (e.g. 12 for Optical, 2 for SAR)
        self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=0)
        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "avg":
            return super().forward(x)
        return self.features(x)

# =========================================================================
# GETSITS ENCODER INTEGRATION
# =========================================================================

class MMFusion(Encoder):
    def __init__(
        self,
        model_name: str,
        input_bands: dict,
        output_layers: list,
        output_dim: list,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        input_size: int = 224,
        multi_temporal: bool = False,
        multi_temporal_output: bool = False,
        pyramid_output: bool = False,
        modalities_finetune: dict = None,
        encoder_weights: str = "",
        download_url: str = "",
        positional_encoding: str = "normal",
        projection_dim: int = 256,
        fusion: str = "concat",
        dropout: float = 0.5,
        pool: str = "cls",
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=multi_temporal_output,
            pyramid_output=pyramid_output,
            encoder_weights=encoder_weights,
            download_url=download_url,
            positional_encoding=positional_encoding
        )
        
        self.modalities = list(input_bands.keys())
        self.topology = [output_dim for _ in self.output_layers]
        self.pool = pool
        self.fusion = fusion
        
        # H' and W' resulting from AlexNet with 224x224 input is 6x6.
        self.feat_size = 6
        
        self.encoders = nn.ModuleDict()
        self.tmaps = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        
        for mod, bands in input_bands.items():
            in_channels = len(bands)
            
            # 1. Modality Encoder
            self.encoders[mod] = AlexNetEncoder(
                in_channels=in_channels,
                latent_dim=256, 
                dropout=dropout, 
                global_pool=""
            )
            
            # 2. Temporal Aggregation
            self.tmaps[mod] = LTAE2d(
                in_channels=256, # AlexNet features output channels
                d_model=256,
                n_head=8,
                mlp=[256, 256],
                return_att=True,
                d_k=4,
                positional_encoding=positional_encoding,
                layer_norm=True
            )
            
            # 3. Input Adapters
            self.adapters[mod] = PatchedInputAdapter(
                num_channels=256,
                stride_level=1,
                patch_size_full=1,
                dim_tokens=embed_dim,
                image_size=self.feat_size
            )

        # 4. Fusion Transformer
        self.fusion_transformer = FusionTransformer(
            width=embed_dim,
            n_heads=num_heads,
            n_layers=depth,
            fusion=fusion,
            pool=pool,
            add_bias_kv=False,
            dropout=0.,
            batch_first=True
        )

        self.projector = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", nn.Flatten(1)),
            ("layer1", nn.Linear(embed_dim, embed_dim)),
            ("bn1", nn.SyncBatchNorm(embed_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(embed_dim, embed_dim)),
            ("bn2", nn.SyncBatchNorm(embed_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(embed_dim, projection_dim)),
        ]))

        self.modalities_finetune = modalities_finetune if modalities_finetune is not None else {}
        self._freeze_modalities()

    def _freeze_modalities(self):
        for mod in self.modalities:
            if not self.modalities_finetune.get(mod, True):
                for param in self.encoders[mod].parameters():
                    param.requires_grad = False
                for param in self.tmaps[mod].parameters():
                    param.requires_grad = False

    def load_encoder_weights(self, logger, from_scratch=False):
        if not from_scratch and self.encoder_weights:
            logger.info(f"Loading pre-trained weights from {self.encoder_weights}...")
            model_dict = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)["model"]
            filtered_dict = {k: v for k, v in model_dict.items() if "projector" not in k}
            self.load_state_dict(filtered_dict, strict=False)
            logger.info("Pre-trained weights loaded successfully.")

    def forward(self, x: dict, batch_positions=None, return_projected=False, mask_modalities=None):
        latent_tokens = []
        
        for mod in self.modalities:
            mod_x = x[mod]
            
            # Metadata handling for LTAE
            mod_bp = None
            if batch_positions is not None:
                mod_bp = {k: v for k, v in batch_positions.items() if not isinstance(v, dict)}
                if mod in batch_positions and isinstance(batch_positions[mod], dict):
                    mod_bp.update(batch_positions[mod])
                if not mod_bp:
                    mod_bp = batch_positions
            
            if mod_x.dim() == 4:
                mod_x = mod_x.unsqueeze(2)
                
            B, C, T, H, W = mod_x.shape
            
            mod_x = mod_x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            features = self.encoders[mod](mod_x) # Output: [B*T, 256, 6, 6]
            
            _, C_f, H_f, W_f = features.shape
            features = features.view(B, T, C_f, H_f, W_f).permute(0, 2, 1, 3, 4)
            temporal_fused, _ = self.tmaps[mod](features, mod_bp) # Output: [B, 256, 6, 6]
            
            tokens = self.adapters[mod](temporal_fused) # Output: [B, 36, embed_dim]
            latent_tokens.append(tokens)

        # --- Lógica de Enmascaramiento ---
        if mask_modalities is None:
            # Si no hay máscara, el comportamiento por defecto es "todas activas"
            list_mask_mod = [len(self.modalities) * [True]]
        else:
            list_mask_mod = mask_modalities

        all_z_projected = []
        final_spatial_list = []
        
        # Iteramos sobre cada máscara (ej: solo-óptico, solo-sar, ambas)
        for mask_mod in list_mask_mod:
            # Filtramos los tokens usando la máscara
            latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
            
            # Obtenemos la cantidad de modalidades activas en ESTA pasada
            active_mods = sum(mask_mod)
            
            layer_features = self.fusion_transformer(latent_tokens_, output_layers=self.output_layers)
            
            out_spatial_list = []
            
            for f in layer_features:
                B_f, Seq_len, D_f = f.shape
                
                # Removemos el token CLS si existe
                start_idx = 1 if (self.fusion == "concat" and self.pool == "cls") else 0
                f_no_cls = f[:, start_idx:, :]
                
                # Dividimos la secuencia por las modalidades ACTIVAS
                tokens_per_mod = (Seq_len - start_idx) // active_mods
                
                # Reconstruimos los mapas espaciales combinando/sumando
                spatial_map = 0
                for i in range(active_mods):
                    mod_tokens = f_no_cls[:, i*tokens_per_mod : (i+1)*tokens_per_mod, :]
                    # Volvemos a la dimensionalidad 6x6 (H_f, W_f)
                    mod_map = mod_tokens.transpose(1, 2).reshape(B_f, D_f, self.feat_size, self.feat_size)
                    spatial_map = spatial_map + mod_map
                    
                out_spatial_list.append(spatial_map)
            
            if return_projected:
                # Proyectamos la última capa y guardamos el vector
                all_z_projected.append(self.projector(out_spatial_list[-1]))
            else:
                # Si no estamos en preentrenamiento contrastivo, guardamos la lista espacial
                final_spatial_list = out_spatial_list

        # --- Retorno final ---
        if return_projected:
            # Si hay más de 1 proyección (por las máscaras), devolvemos la lista completa
            return all_z_projected if len(all_z_projected) > 1 else all_z_projected[0]
            
        # Si es para segmentación/finetuning, devolvemos la pirámide de mapas 2D sumada
        return final_spatial_list