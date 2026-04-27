import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from einops import rearrange, repeat
from .base import Encoder as BaseEncoder
import math


class ModalityConfig:
    def __init__(self, name, bands, image_tile_size_factor=1, is_spatial=True, is_multitemporal=True):
        self.name = name
        self.bands = bands
        self.image_tile_size_factor = image_tile_size_factor
        self.is_spatial = is_spatial
        self.is_multitemporal = is_multitemporal
        self.num_band_sets = len(bands)

    def bandsets_as_indices(self):
        indices = []
        start = 0
        for b in self.bands:
            indices.append(list(range(start, start + len(b))))
            start += len(b)
        return indices

MODALITIES = {
    "sentinel2_l2a": ModalityConfig("sentinel2_l2a", [
        ["B02", "B03", "B04", "B08"],               # 10m
        ["B05", "B06", "B07", "B8A", "B11", "B12"], # 20m
        ["B01", "B09"]                              # 60m
    ], image_tile_size_factor=1)
}

BASE_GSD = 10.0  # Ground Sample Distance base


def get_1d_sincos_pos_encoding(pos: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    assert encoding_dim % 2 == 0
    omega = torch.arange(encoding_dim // 2, device=pos.device).float() / encoding_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1).float()
    out = torch.einsum("l,d->ld", pos, omega)
    
    encoding = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return encoding

def get_2d_sincos_pos_encoding(grid: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    assert encoding_dim % 2 == 0
    encoding_dim_1d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], encoding_dim_1d)
    emb_w = get_1d_sincos_pos_encoding(grid[1], encoding_dim_1d)
    return torch.cat([emb_h, emb_w], dim=1)

def get_2d_sincos_pos_encoding_with_resolution(
    grid_size: int,
    res: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
) -> torch.Tensor:
    grid_h = torch.arange(grid_size, device=device).float()
    grid_w = torch.arange(grid_size, device=device).float()
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0) # 2 x h x w

    # Escalar por resolución
    grid = torch.einsum("chw,n->cnhw", grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)
    return pos_embed

def get_month_encoding_table(encoding_dim: int) -> torch.Tensor:
    """Genera tabla fija de embeddings para 12 meses + 1 (idx 13 reservado)"""
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13).float() / (12 / (2 * np.pi))
    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    return torch.cat([sin_table[:-1], cos_table[:-1]], dim=-1)

class FlexiPatchEmbed(nn.Module):
    def __init__(self, in_chans, embedding_size, patch_size_at_16, image_tile_size_factor=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_tile_size_factor = image_tile_size_factor
        # Patch size efectivo
        self.patch_size = (patch_size_at_16 * image_tile_size_factor, patch_size_at_16 * image_tile_size_factor)
        
        self.proj = nn.Conv2d(
            in_chans, embedding_size, kernel_size=self.patch_size, stride=self.patch_size, bias=True
        )

    def forward(self, x: torch.Tensor, patch_size: int = None) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x) # [B, Embed, H', W']
        x = rearrange(x, "b d h w -> b h w d")
        return x

class MultiModalPatchEmbeddings(nn.Module):
    def __init__(self, supported_modalities, max_patch_size, embedding_size):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.per_modality_embeddings = nn.ModuleDict()
        
        for modality_name in supported_modalities:
            mod_config = MODALITIES[modality_name]
            for idx, band_indices in enumerate(mod_config.bandsets_as_indices()):
                embed_key = f"{modality_name}__{idx}"
                self.per_modality_embeddings[embed_key] = FlexiPatchEmbed(
                    in_chans=len(band_indices),
                    embedding_size=embedding_size,
                    patch_size_at_16=max_patch_size,
                    image_tile_size_factor=mod_config.image_tile_size_factor
                )
                buffer_name = f"{modality_name}__{idx}_buffer"
                self.register_buffer(buffer_name, torch.tensor(band_indices, dtype=torch.long), persistent=False)

    def forward(self, x_dict: Dict[str, torch.Tensor], patch_size: int) -> Dict[str, torch.Tensor]:
        output = {}
        for mod_name, x in x_dict.items():
            if mod_name not in MODALITIES: continue
            
            # x: [B, H, W, T, C_total] -> [B*T, C_total, H, W]
            b, h, w, t, c = x.shape
            x_reshaped = rearrange(x, "b h w t c -> (b t) c h w")
            
            mod_tokens = []
            mod_config = MODALITIES[mod_name]
            
            for idx in range(mod_config.num_band_sets):
                indices = getattr(self, f"{mod_name}__{idx}_buffer")
                x_subset = torch.index_select(x_reshaped, 1, indices)
                
                embed_module = self.per_modality_embeddings[f"{mod_name}__{idx}"]
                tokens = embed_module(x_subset, patch_size=patch_size) # [(B T), H', W', D]
                mod_tokens.append(tokens)
            
            tokens_stacked = torch.stack(mod_tokens, dim=-2)
            
            _, h_prime, w_prime, _, _ = tokens_stacked.shape
            tokens_stacked = rearrange(tokens_stacked, "(b t) h w bs d -> b h w t bs d", b=b, t=t)
            
            output[mod_name] = tokens_stacked
        return output

class CompositeEncodings(nn.Module):
    def __init__(self, embedding_size, supported_modalities, max_sequence_length=12):
        super().__init__()
        self.embedding_dim_per_type = int(embedding_size * 0.25)
        
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(torch.arange(max_sequence_length), self.embedding_dim_per_type),
            requires_grad=False
        )
        
        month_tab = get_month_encoding_table(self.embedding_dim_per_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        
        self.per_modality_channel_embeddings = nn.ParameterDict()
        for mod_name in supported_modalities:
            num_sets = MODALITIES[mod_name].num_band_sets
            self.per_modality_channel_embeddings[mod_name] = nn.Parameter(
                torch.zeros(num_sets, self.embedding_dim_per_type)
            )
            
    def calculate_gsd_ratio(self, input_res, patch_size):
        return input_res * patch_size / BASE_GSD

    def forward(self, tokens_dict, timestamps, patch_size, input_res=10.0):
        # tokens_dict[mod]: [B, H, W, T, BandSet, D]
        # timestamps: [B, T, 2] -> [Year, MonthIndex]
        output = {}
        n = self.embedding_dim_per_type
        
        for mod_name, tokens in tokens_dict.items():
            b, h, w, t, bs, d = tokens.shape
            device = tokens.device
            composite_embed = torch.zeros_like(tokens) 
            
            # A. Channel Embedding (0:25%)
            chan_embed = self.per_modality_channel_embeddings[mod_name]
            chan_embed = repeat(chan_embed, "bs d -> b h w t bs d", b=b, h=h, w=w, t=t)
            composite_embed[..., :n] += chan_embed
            
            # B. Time Sequence Embedding (25:50%)
            safe_t = min(t, self.pos_embed.shape[0])
            time_embed = self.pos_embed[:safe_t] 
            
            if t > safe_t:
                 last_emb = time_embed[-1:]
                 padding = repeat(last_emb, "1 d -> (t_diff) d", t_diff=t-safe_t)
                 time_embed = torch.cat([time_embed, padding], dim=0)

            time_embed = repeat(time_embed, "t d -> b h w t bs d", b=b, h=h, w=w, bs=bs)
            composite_embed[..., n : n*2] += time_embed
            
            # C. Month Embedding (50:75%)
            if timestamps is not None:
                months = timestamps[:, :, 1].long()
                month_emb = self.month_embed(months)
                month_emb = repeat(month_emb, "b t d -> b h w t bs d", h=h, w=w, bs=bs)
                composite_embed[..., n*2 : n*3] += month_emb
                
            # D. Spatial Embedding (75:100%)
            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)
            spatial_emb = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=h, 
                res=torch.ones(b, device=device) * gsd_ratio,
                encoding_dim=n,
                device=device
            )
            spatial_emb = rearrange(spatial_emb, "b (h w) d -> b h w d", h=h, w=w)
            spatial_emb = repeat(spatial_emb, "b h w d -> b h w t bs d", t=t, bs=bs)
            composite_embed[..., n*3 : ] += spatial_emb
            
            output[mod_name] = tokens + composite_embed
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ProjectAndAggregate(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(embedding_size, embedding_size))
    def forward(self, x):
        return self.projection(x)

class OlmoEarthEncoderBackbone(nn.Module):
    def __init__(
        self,
        embedding_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        max_patch_size=8,
        supported_modalities=["sentinel2_l2a"]
    ):
        super().__init__()
        self.supported_modalities = supported_modalities
        
        # 1. Patch Embeddings
        self.patch_embeddings = MultiModalPatchEmbeddings(
            supported_modalities, max_patch_size, embedding_size
        )

        # 2. Composite Encodings
        self.composite_encodings = CompositeEncodings(
            embedding_size, supported_modalities, max_sequence_length=12
        )

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embedding_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True, 
                norm_layer=nn.LayerNorm
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embedding_size)
        self.project_and_aggregate = ProjectAndAggregate(embedding_size)

    def forward_features(self, x_dict: Dict[str, torch.Tensor], timestamps: torch.Tensor, patch_size: int, out_indices: List[int]) -> List[torch.Tensor]:
        # 1. Patchify -> [B, H_grid, W_grid, T, BandSet, D]
        tokens_dict = self.patch_embeddings(x_dict, patch_size=patch_size)
        
        # 2. Add Composite Encodings
        tokens_dict = self.composite_encodings(tokens_dict, timestamps, patch_size)
        
        # 3. Flatten Sequence
        modality = self.supported_modalities[0] # Asumimos S2
        x = tokens_dict[modality] 
        B, H, W, T, BS, D = x.shape
        x_flat = rearrange(x, "b h w t bs d -> b (h w t bs) d")
        
        features = []
        for i, block in enumerate(self.blocks):
            x_flat = block(x_flat)
            
            if i in out_indices:
                x_rec = rearrange(x_flat, "b (h w t bs) d -> b h w t bs d", h=H, w=W, t=T, bs=BS)
                
                x_pool = torch.mean(x_rec, dim=(3, 4)) 
                
                feat = x_pool.permute(0, 3, 1, 2).contiguous()
                features.append(feat)
                
        return features
    
    def interpolate_pos_embed(self, old_pos_embed, new_shape):
        cls_token = old_pos_embed[:, 0:1, :]
        grid_tokens = old_pos_embed[:, 1:, :]
        
        target_grid_len = new_shape[1] - 1 
        current_grid_len = grid_tokens.shape[1]
        
        if target_grid_len == current_grid_len:
            return old_pos_embed
            
        size_old = int(math.sqrt(current_grid_len))
        size_new = int(math.sqrt(target_grid_len))
        
        if size_old * size_old != current_grid_len:
            return old_pos_embed 
            
        grid_tokens = rearrange(grid_tokens, 'b (h w) d -> b d h w', h=size_old, w=size_old)
        
        new_grid_tokens = F.interpolate(
            grid_tokens, 
            size=(size_new, size_new), 
            mode='bicubic', 
            align_corners=False
        )
        
        new_grid_tokens = rearrange(new_grid_tokens, 'b d h w -> b (h w) d')
        
        return torch.cat((cls_token, new_grid_tokens), dim=1)

    def load_pretrained_weights(self, state_dict):
        new_state_dict = {}
        own_state = self.state_dict()
        
        mismatched_shapes = []
        interpolated_keys = []

        for k, v in state_dict.items():
            if k.startswith("encoder."): new_key = k.replace("encoder.", "")
            elif k.startswith("backbone."): new_key = k.replace("backbone.", "")
            elif k.startswith("decoder"): continue 
            else: new_key = k

            if "per_modality_embeddings" in new_key:
                for mod in self.supported_modalities:
                    nested_pattern = f".{mod}.{mod}__"
                    flat_pattern = f".{mod}__"
                    if nested_pattern in new_key:
                        new_key = new_key.replace(nested_pattern, flat_pattern)
            
            if new_key in own_state:
                target_shape = own_state[new_key].shape
                if v.shape != target_shape:
                    
                    if "pos_embed" in new_key and len(v.shape) == 3:
                        v = self.interpolate_pos_embed(v, target_shape)
                        if v.shape == target_shape:
                            interpolated_keys.append(new_key)
                        
                    elif "patch_embed" in new_key and "proj.weight" in new_key and len(v.shape) == 4:
                        v = F.interpolate(
                            v, 
                            size=(target_shape[2], target_shape[3]), 
                            mode='bicubic', 
                            align_corners=False
                        )
                        interpolated_keys.append(new_key)
                    
                    if v.shape != target_shape:
                        mismatched_shapes.append({
                            "key": new_key,
                            "pretrained": str(list(v.shape)),
                            "current": str(list(target_shape))
                        })
                        continue 
            
            new_state_dict[new_key] = v
        
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        return missing, unexpected, mismatched_shapes, interpolated_keys

class OlmoEarth(BaseEncoder):
    def __init__(
        self,
        encoder_weights=None,
        model_name="olmo_earth",
        input_size=224,
        input_bands=None, 
        embed_dim=768,
        output_layers=[3, 5, 7, 11],
        output_dim=768,
        download_url=None,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,  
        projection_dim: int = 64, 
        positional_encoding: str | None = "normal", 
    ):
        super().__init__(
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
            positional_encoding=positional_encoding
        )
        
        self.topology = [output_dim for _ in self.output_layers]
        
        self.encoder_weights = encoder_weights
        self.output_layers = output_layers
        self.patch_size = patch_size
        
        self.band_mapping = {
            "group_0_idx": [1, 2, 3, 7],            # 10m: B2, B3, B4, B8
            "group_1_idx": [4, 5, 6, 8, 10, 11],    # 20m: B5, B6, B7, B8A, B11, B12
            "group_2_idx": [0, 9]                   # 60m: B1, B9
        }
        
        self.backbone = OlmoEarthEncoderBackbone(
            embedding_size=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_patch_size=patch_size, 
            supported_modalities=["sentinel2_l2a"]
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

    def forward(self, x, batch_positions=None):
        """
        Args:
            x: Tensor [B, C, T, H, W]
            batch_positions: Tensor [B, T] (DOY 1-366 normalized)
        """
        x=x.permute(0,2,1,3,4)
        B, T, C, H, W = x.shape
        
        pad_mask = (x == 0).all(dim=2).all(dim=2).all(dim=2) # [B, T]

        g0 = x[:, :, self.band_mapping["group_0_idx"], :, :] # [B, T, 4, H, W]
        g1 = x[:, :, self.band_mapping["group_1_idx"], :, :] # [B, T, 6, H, W]
        g2 = x[:, :, self.band_mapping["group_2_idx"], :, :] # [B, T, 2, H, W]
        
        x_ordered = torch.cat([g0, g1, g2], dim=2) # [B, T, 12, H, W]
        
        x_input = x_ordered.permute(0, 3, 4, 1, 2) 
        x_dict = {"sentinel2_l2a": x_input}
        
        if batch_positions is not None:
            months = ((batch_positions["doy"]*365 - 1) / 30.5).long().clamp(0, 11)
            years = torch.full_like(months, 2023) # Año dummy
            timestamps = torch.stack([years, months], dim=-1) # [B, T, 2]
        else:
            timestamps = torch.zeros(B, T, 2, dtype=torch.long, device=x.device)

        features = self.backbone.forward_features(
            x_dict, 
            timestamps=timestamps, 
            patch_size=self.patch_size, 
            out_indices=self.output_layers
        )
        
        out = features[-1] 
        
        return out, features, pad_mask, None

    def load_encoder_weights(self, logger=None, from_scratch=False):
        if from_scratch or self.encoder_weights is None:
            return

        if logger: logger.info(f"Loading OlmoEarth weights from {self.encoder_weights}...")
        
        try:
            state_dict = torch.load(self.encoder_weights, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            missing, _, _, interpd = self.backbone.load_pretrained_weights(state_dict)
            
            if logger:
                logger.info("OlmoEarth weights loaded successfully.")
                logger.info(f"Missing Params: {missing}")
                logger.info(f"Adapted Params: {interpd}")
            
        except Exception as e:
            if logger: logger.error(f"Error loading weights: {e}")
            raise e
    
    def __str__(self):
        return "OlmoEarth"
