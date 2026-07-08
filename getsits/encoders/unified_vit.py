import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

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
        out_i = self.conv_i(x)
        out_ii = self.conv_ii(out_i)
        
        cat_1 = torch.cat([out_i, out_ii], dim=1)
        out_skip1 = self.skip1(cat_1)
        
        out_iii = self.conv_iii(out_skip1)
        
        cat_2 = torch.cat([out_skip1, out_iii], dim=1)
        out_skip2 = self.skip2(cat_2)
        
        return self.patching(out_skip2)


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
        self.modality_embeds = nn.ParameterDict()
        self.tmaps = nn.ModuleDict()
        
        for mod, bands in input_bands.items():
            self.tokenizers[mod] = DenseConvStem(len(bands), embed_dim, patch_size)
            self.modality_embeds[mod] = nn.Parameter(torch.zeros(1, 1, embed_dim))
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

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

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

        self.modalities_finetune = modalities_finetune if modalities_finetune is not None else {}
        self._freeze_modalities()
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        for mod in self.modalities:
            nn.init.trunc_normal_(self.modality_embeds[mod], std=.02)

    def _freeze_modalities(self):
        for mod in self.modalities:
            if not self.modalities_finetune.get(mod, True):
                for param in self.tokenizers[mod].parameters():
                    param.requires_grad = False
                self.modality_embeds[mod].requires_grad = False
                for param in self.tmaps[mod].parameters():
                    param.requires_grad = False

    def load_encoder_weights(self, logger, from_scratch=False):
        if not from_scratch and self.encoder_weights:
            checkpoint = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights for {self.model_name} from {self.encoder_weights}")

    def forward(self, x: dict, batch_positions=None, force_alpha=None, return_projected=False):
        processed_tokens = {}
        doy = batch_positions["doy"] if batch_positions is not None else None
        
        for mod in self.modalities:
            mod_x = x[mod]
            
            if mod_x.dim() == 4:
                mod_x = mod_x.unsqueeze(2)
                
            B, C, T, H, W = mod_x.shape
            mod_x = mod_x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            toks_spatial = self.tokenizers[mod](mod_x)
            H_p, W_p = toks_spatial.shape[-2:]
            
            toks_spatial = toks_spatial.view(B, T, self.embed_dim, H_p, W_p).permute(0, 2, 1, 3, 4)
            temporal_fused, _ = self.tmaps[mod](toks_spatial, doy)
            
            toks_flat = temporal_fused.flatten(2).transpose(1, 2)
            toks_flat = toks_flat + self.pos_embed + self.modality_embeds[mod]
            processed_tokens[mod] = toks_flat

        # Manejo matemático de la Modalidad: fuerza a Óptico(1.0), SAR(0.0), Fusión(0.5) o Estocástico
        fused_tokens = torch.zeros_like(processed_tokens[self.modalities[0]])
        
        if force_alpha is not None:
            fused_tokens = force_alpha * processed_tokens["optical"] + (1 - force_alpha) * processed_tokens["sar"]
        elif self.training:
            B = fused_tokens.shape[0]
            rand_val = torch.rand(B, 1, 1, device=fused_tokens.device)
            alpha = torch.zeros_like(rand_val)
            alpha[rand_val < 0.33] = 1.0 
            alpha[(rand_val >= 0.33) & (rand_val < 0.66)] = 0.0 
            
            mix_mask = rand_val >= 0.66
            alpha[mix_mask] = torch.rand(mix_mask.sum(), 1, 1, device=fused_tokens.device)
            fused_tokens = alpha * processed_tokens["optical"] + (1 - alpha) * processed_tokens["sar"]
        else:
            fused_tokens = sum(processed_tokens.values()) / len(self.modalities)

        out = fused_tokens
        for blk in self.blocks:
            out = blk(out)
        
        out = self.norm(out)
        
        B, N, D = out.shape
        out_spatial = out.transpose(1, 2).reshape(B, D, H_p, W_p)
        
        if return_projected:
            return self.projector(out_spatial)
            
        return [out_spatial]