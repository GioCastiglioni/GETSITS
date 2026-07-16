import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint

from getsits.encoders.base import Encoder
from getsits.encoders.ltae import LTAE2d

class DenseConvStem(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.conv_ii = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.conv_iii = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.patching = nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner_forward(x_in):
            out_i = self.conv_i(x_in)
            out_ii = self.conv_ii(out_i)
            
            cat_1 = torch.cat([out_i, out_ii], dim=1)
            out_skip1 = self.skip1(cat_1)
            
            out_iii = self.conv_iii(out_skip1)
            
            cat_2 = torch.cat([out_skip1, out_iii], dim=1)
            out_skip2 = self.skip2(cat_2)
            
            return self.patching(out_skip2)
            
        if self.training:
            return checkpoint(_inner_forward, x, use_reentrant=False)
        return _inner_forward(x)

class AttentionStem(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16, input_size: int = 224, depth: int = 1, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=2.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_T, C, H, W = x.shape
        
        # [B*T, D, H/P, W/P]
        x = self.proj(x)
        H_p, W_p = x.shape[-2:]
        
        # [B*T, N, D]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            
        x = self.norm(x)
        
        # [B*T, D, H/P, W/P]
        x = x.transpose(1, 2).reshape(B_T, -1, H_p, W_p)
        return x


class UnifiedMMViT(Encoder):
    def __init__(
        self,
        model_name: str,
        input_bands: dict,
        output_layers: list,
        output_dim: list,
        embed_dim: int = 384,
        patch_size: int = 16,
        depth: int = 12,
        num_heads: int = 6,
        input_size: int = 224,
        multi_temporal: bool = False,
        multi_temporal_output: bool = False,
        pyramid_output: bool = False,
        modalities_finetune: dict = None,
        encoder_weights: str = "",
        download_url: str = "",
        positional_encoding: str = "normal",
        projection_dim: int = 256,
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
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        self.topology = [output_dim for _ in self.output_layers]
        
        self.tokenizers = nn.ModuleDict()
        self.tmaps = nn.ModuleDict()
        
        for mod, bands in input_bands.items():
            #self.tokenizers[mod] = DenseConvStem(len(bands), embed_dim, patch_size)
            self.tokenizers[mod] = AttentionStem(
                in_channels=len(bands), 
                embed_dim=embed_dim, 
                patch_size=patch_size,
                input_size=input_size,
                depth=1,
                num_heads=4
            )
            self.tmaps[mod] = LTAE2d(
            in_channels=self.topology[-1],
            d_model=256,
            n_head=16,
            mlp=[256, self.topology[-1]],
            return_att=True,
            d_k=4,
            positional_encoding=positional_encoding,
            layer_norm=True
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * len(self.modalities), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.topology[-1], embed_dim),
            nn.SyncBatchNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.SyncBatchNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, projection_dim)
        )

        self.modalities_finetune = modalities_finetune if modalities_finetune is not None else {}
        self._freeze_modalities()
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def _freeze_modalities(self):
        for mod in self.modalities:
            if not self.modalities_finetune.get(mod, True):
                for param in self.tokenizers[mod].parameters():
                    param.requires_grad = False
                for param in self.tmaps[mod].parameters():
                    param.requires_grad = False

    def load_encoder_weights(self, logger, from_scratch=False):
        if not from_scratch:
            logger.info(f"Loading pre-trained weights from {self.encoder_weights}...")
            model_dict = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)["model"]
            self.load_state_dict(model_dict)
            logger.info("Pre-trained weights loaded successfully.")
        else:pass

    def forward(self, x: dict, batch_positions=None, force_alpha=None, return_projected=False):
        processed_tokens = {}
        
        for mod in self.modalities:
            mod_x = x[mod]
            
            # 1. Extraer los metadatos: combinando los globales (lat, lon) con los locales (doy, time)
            mod_bp = None
            if batch_positions is not None:
                # Rescata todo lo que no sea un diccionario (ej. "lat", "lon")
                mod_bp = {k: v for k, v in batch_positions.items() if not isinstance(v, dict)}
                
                # Suma los datos específicos de la modalidad (ej. "doy" de optical)
                if mod in batch_positions and isinstance(batch_positions[mod], dict):
                    mod_bp.update(batch_positions[mod])
                    
                # Si el dataset era antiguo y plano, usamos todo
                if not mod_bp:
                    mod_bp = batch_positions
            
            if mod_x.dim() == 4:
                mod_x = mod_x.unsqueeze(2)
                
            B, C, T, H, W = mod_x.shape
            mod_x = mod_x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            toks_spatial = self.tokenizers[mod](mod_x)
            H_p, W_p = toks_spatial.shape[-2:]
            toks_spatial = toks_spatial.view(B, T, self.embed_dim, H_p, W_p).permute(0, 2, 1, 3, 4)
            temporal_fused, _ = self.tmaps[mod](toks_spatial, mod_bp)
            
            toks_flat = temporal_fused.flatten(2).transpose(1, 2)
            
            # Fiel a CoMM: Solo se suma la codificación posicional espacial, NO modality_embeds
            toks_flat = toks_flat + self.pos_embed 
            processed_tokens[mod] = toks_flat

        concat_tokens = torch.cat([processed_tokens["optical"], processed_tokens["sar"]], dim=-1)
        fused_tokens = self.fusion_proj(concat_tokens)

        out = fused_tokens
        features = []
        
        for i, blk in enumerate(self.blocks):
            if self.training:
                out = checkpoint(blk, out, use_reentrant=False)
            else:
                out = blk(out)
            
            if i in self.output_layers:
                features.append(self.norm(out))
        
        out_spatial_list = []
        for f in features:
            B, N, D = f.shape
            out_spatial_list.append(f.transpose(1, 2).reshape(B, D, H_p, W_p))
        
        if return_projected:
            return self.projector(out_spatial_list[-1])
            
        return out_spatial_list