# Based on the original implementation: https://github.com/nasaharvest/galileo

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from einops import rearrange, repeat
from .base import Encoder as BaseEncoder
import collections
import itertools

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]

SPACE_TIME_BANDS_GROUPS_IDX = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_month_encoding_table(embed_dim):
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))
    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
    return month_table

def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, device="cpu"):
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = torch.einsum("chw,n->cnhw", grid, res)
    _, n, h, w = grid.shape
    
    def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    return pos_embed

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class FlexiPatchEmbed(nn.Module):
    def __init__(
        self, 
        patch_size, 
        in_chans=3, 
        embed_dim=128, 
        norm_layer=None, 
        bias=True,
        patch_size_seq=(8, 10, 12, 16, 20, 24, 32), # Secuencia original de entrenamiento
        interpolation="bicubic",
        antialias=True
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias
        self.patch_size_seq = patch_size_seq
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        pinvs = {}
        # Pre-calcula las matrices de pseudo-inversa para los tamaños comunes
        if self.patch_size_seq:
            for ps in self.patch_size_seq:
                tuple_ps = to_2tuple(ps)
                pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x, shape):
        # Utilidad para cambiar tamaño de la base vectorial
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape, new_shape):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed, new_patch_size):
        if self.patch_size == new_patch_size:
            return patch_embed

        # Si el tamaño solicitado no está en caché, calcúlalo al vuelo
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        # Proyección vectorizada de los pesos usando pinv
        def resample_patch_embed(patch_embed):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(self, x, patch_size=None):
        # x input: [B, H, W, (T), C] -> Manejo de dims
        if len(x.shape) == 5:
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)
        x = rearrange(x, "b c h w -> b h w c") # Devuelve shape Transformer
            
        x = self.norm(x) # Original Galileo tiene norm aquí
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class GalileoBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        depth=2,
        num_heads=8,
        mlp_ratio=2,
        max_sequence_length=24,
        patch_size=16,
        use_embed_norm=False, 
        use_qk_norm=False,
        use_layerscale=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX

        embed_norm_layer = nn.LayerNorm if use_embed_norm else nn.Identity
        
        self.space_time_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embed_dim, patch_size=patch_size, norm_layer=embed_norm_layer)
            for group_name, group in self.space_time_groups.items()
        })
        
        self.space_embed = nn.ModuleDict({
            k: FlexiPatchEmbed(in_chans=len(v), embed_dim=embed_dim, patch_size=patch_size, norm_layer=embed_norm_layer)
            for k, v in {"SRTM": [0,1], "DW": list(range(9)), "WC": list(range(5))}.items()
        })
        
        self.time_embed = nn.ModuleDict({
             k: nn.Linear(len(v), embed_dim) for k, v in {"ERA5": [0,1], "TC": [0,1,2], "VIIRS": [0]}.items()
        })
        self.static_embed = nn.ModuleDict({
            k: nn.Linear(len(v), embed_dim) for k, v in {"LS": [0], "location": [0,1,2], "DW_static": list(range(9)), "WC_static": list(range(5))}.items()
        })

        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(int(embed_dim * 0.25), torch.arange(max_sequence_length)), requires_grad=False
        )
        month_tab = get_month_encoding_table(int(embed_dim * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

        self.s_t_channel_embed = nn.Parameter(torch.zeros(len(self.space_time_groups), int(embed_dim * 0.25)))
        self.sp_channel_embed = nn.Parameter(torch.zeros(3, int(embed_dim * 0.25)))
        self.t_channel_embed = nn.Parameter(torch.zeros(3, int(embed_dim * 0.25))) 
        self.st_channel_embed = nn.Parameter(torch.zeros(4, int(embed_dim * 0.25)))

        ls_init_values = 1e-5 if use_layerscale else None

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=True, qk_norm=use_qk_norm,
                norm_layer=nn.LayerNorm, drop_path=0.0, init_values=ls_init_values
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x, months, input_res=10.0, output_indices=None, patch_size=None):
        # x shape: [B, H, W, T, C_total]
        b, h, w, t, _ = x.shape
        current_patch_size = patch_size if patch_size is not None else self.patch_size
        
        s_t_tokens = []
        for idx, (group_name, channel_idxs) in enumerate(self.space_time_groups.items()):
            group_x = x[..., channel_idxs]
            # FlexiPatchEmbed handles internal norm and resizing
            tokens = self.space_time_embed[group_name](group_x, patch_size=current_patch_size)
            s_t_tokens.append(tokens)
            
        s_t_tokens = torch.stack(s_t_tokens, dim=-2)
        s_t_tokens = rearrange(s_t_tokens, "(b t) h w g d -> b h w t g d", b=b, t=t)
        
        new_h, new_w = s_t_tokens.shape[1], s_t_tokens.shape[2]
        dim_part = int(self.embed_dim * 0.25)
        
        c_emb = repeat(self.s_t_channel_embed, "g d -> b h w t g d", b=b, h=new_h, w=new_w, t=t)
        
        safe_t = min(t, self.pos_embed.shape[0])
        p_emb = self.pos_embed[:safe_t]
        if t > safe_t:
             p_emb = torch.cat([p_emb, repeat(p_emb[-1:], "1 d -> k d", k=t-safe_t)], dim=0)
        p_emb = repeat(p_emb, "t d -> b h w t g d", b=b, h=new_h, w=new_w, g=len(self.space_time_groups))
        
        m_emb = self.month_embed(months) 
        m_emb = repeat(m_emb, "b t d -> b h w t g d", h=new_h, w=new_w, g=len(self.space_time_groups))
        
        token_res = input_res * current_patch_size
        gsd_ratio = token_res / 10.0 
        sp_emb = get_2d_sincos_pos_embed_with_resolution(
            dim_part, new_h, torch.ones(b, device=x.device) * gsd_ratio, device=x.device
        ) 
        sp_emb = rearrange(sp_emb, "b (h w) d -> b h w d", h=new_h, w=new_w)
        sp_emb = repeat(sp_emb, "b h w d -> b h w t g d", t=t, g=len(self.space_time_groups))
        
        total_emb = torch.cat([c_emb, p_emb, m_emb, sp_emb], dim=-1)
        
        x = s_t_tokens + total_emb
        x = rearrange(x, "b h w t g d -> b (h w t g) d")
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            if output_indices and i in output_indices:
                curr_x = x
                if i == len(self.blocks) - 1:
                    curr_x = self.norm(x)
                
                x_rec = rearrange(curr_x, "b (h w t g) d -> b h w t g d", h=new_h, w=new_w, t=t)
                
                x_pool = torch.mean(x_rec, dim=(3, 4)) 
                feat = x_pool.permute(0, 3, 1, 2).contiguous() 
                features.append(feat)
             
        return features

class GalileoTiny(BaseEncoder):
    def __init__(
        self,
        encoder_weights=None,
        model_name="",
        input_size=224,
        input_bands=None,
        embed_dim=128, 
        output_layers=[1],
        output_dim=128,
        download_url=None,
        patch_size=16,
        pretrained_patch_size=8,
        depth=2,
        num_heads=8,
        mlp_ratio=2,
        projection_dim=64,
        positional_encoding="normal",
        input_scaling="auto"
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
        self.patch_size = patch_size
        self.output_layers = output_layers
        self.input_scaling = input_scaling
        
        self.backbone = GalileoBackbone(
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            patch_size=pretrained_patch_size,
            max_sequence_length=24,
            use_embed_norm=False,
            use_qk_norm=False,
            use_layerscale=False
        )

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dim, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, projection_dim)
        )
        
        self.s2_indices_map = {
            "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, 
            "B7": 6, "B8": 7, "B8A": 8, "B11": 10, "B12": 11
        }
        
        self.input_bands_list = []
        if isinstance(input_bands, (dict, OrderedDict)) or hasattr(input_bands, 'keys'):
            if 'optical' in input_bands:
                self.input_bands_list = list(input_bands['optical'])
            else:
                for modality in input_bands.values():
                    self.input_bands_list.extend(list(modality))
        elif isinstance(input_bands, (list, tuple)):
            self.input_bands_list = list(input_bands)
        
        self.input_band_to_idx = {b: i for i, b in enumerate(self.input_bands_list)}

    def _get_scaling_factor(self, x):
        if isinstance(self.input_scaling, (int, float)):
            return float(self.input_scaling)
        if self.input_scaling == "auto":
            max_val = x.max().item()
            if max_val == 0: return 1.0 
            if max_val > 255.0: return 10000.0
            elif max_val > 1.5: return 255.0
            else: return 1.0
        return 1.0

    def _prepare_input(self, x):
        b, t, c, h, w = x.shape
        device = x.device
        
        scale_factor = self._get_scaling_factor(x)
        x = x.float() / (scale_factor + 1e-6)
        
        # Placeholder S1
        s1 = torch.zeros(b, t, 2, h, w, device=device)
        
        target_s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        s2_bands_list = []
        has_band_map = len(self.input_band_to_idx) > 0

        for b_name in target_s2_bands:
            if b_name in self.input_band_to_idx:
                idx = self.input_band_to_idx[b_name]
                s2_bands_list.append(x[:, :, idx:idx+1, :, :])
            elif not has_band_map and b_name in self.s2_indices_map and self.s2_indices_map[b_name] < c:
                idx = self.s2_indices_map[b_name]
                s2_bands_list.append(x[:, :, idx:idx+1, :, :])
            else:
                s2_bands_list.append(torch.zeros(b, t, 1, h, w, device=device))

        s2 = torch.cat(s2_bands_list, dim=2)
        
        b4, nir = None, None
        if "B4" in self.input_band_to_idx: b4 = x[:, :, self.input_band_to_idx["B4"]:self.input_band_to_idx["B4"]+1, :, :]
        elif not has_band_map and 3 < c: b4 = x[:, :, 3:4, :, :]
            
        if "B8" in self.input_band_to_idx: nir = x[:, :, self.input_band_to_idx["B8"]:self.input_band_to_idx["B8"]+1, :, :]
        elif "B8A" in self.input_band_to_idx: nir = x[:, :, self.input_band_to_idx["B8A"]:self.input_band_to_idx["B8A"]+1, :, :]
        elif not has_band_map and 7 < c: nir = x[:, :, 7:8, :, :]
        
        if b4 is not None and nir is not None:
            ndvi = (nir - b4) / (nir + b4 + 1e-6)
        else:
            ndvi = torch.zeros(b, t, 1, h, w, device=device)
        
        x_galileo = torch.cat([s1, s2, ndvi], dim=2) 
        x_galileo = x_galileo.permute(0, 3, 4, 1, 2) # [B, H, W, T, C]
        return x_galileo

    def forward(self, x, batch_positions=None):
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        pad_mask = (x == 0).all(dim=2).all(dim=2).all(dim=2) 
        
        s_t_x = self._prepare_input(x)
        
        if batch_positions is not None:
            if isinstance(batch_positions, dict): doy = batch_positions.get("doy", torch.zeros(B, T, device=x.device))
            else: doy = batch_positions
            months = ((doy * 365 - 1) / 30.5).long().clamp(0, 11)
        else:
            months = torch.zeros(B, T, dtype=torch.long, device=x.device)

        features = self.backbone.forward_features(
            s_t_x, months=months, input_res=10.0, output_indices=self.output_layers, patch_size=self.patch_size
        )
        
        out = features[-1] if features else None
        return out, features, pad_mask, None

    def load_encoder_weights(self, logger=None, from_scratch=False):
        if from_scratch or self.encoder_weights is None: return
        if logger: logger.info(f"Loading GalileoTiny weights from {self.encoder_weights}...")
        try:
            state_dict = torch.load(self.encoder_weights, map_location="cpu")
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
            elif "model" in state_dict: state_dict = state_dict["model"]
            
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            
            if logger:
                logger.info("GalileoTiny weights loaded.")
                if missing: logger.warning(f"Missing keys: {missing}")
                
        except Exception as e:
            if logger: logger.error(f"Error loading weights: {e}")
            raise e

    def __str__(self):
        return "GalileoTiny"