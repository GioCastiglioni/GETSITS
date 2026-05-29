import torch
import torch.nn as nn
from hydra.utils import instantiate
from getsits.encoders.base import Encoder

class CoMMEncoder(Encoder):
    def __init__(
        self,
        model_name: str,
        input_bands: dict,
        input_size: int,
        embed_dim: int,
        output_layers: list,
        output_dim: list,
        multi_temporal: bool,
        multi_temporal_output: bool,
        pyramid_output: bool,
        encoder_weights: str,
        download_url: str,
        encoders_config: dict,
        num_heads: int = 8,
        depth: int = 1,
        positional_encoding: str = "normal"
    ) -> None:
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
        
        self.encoders = nn.ModuleDict()
        for mod, cfg in encoders_config.items():
            self.encoders[mod] = instantiate(cfg)
        self.modalities = list(self.encoders.keys())
        
        self.latent_converters = nn.ModuleDict()
        for mod in self.modalities:
            enc_out_dim = self.encoders[mod].output_dim[-1]
            self.latent_converters[mod] = nn.Sequential(
                nn.Conv2d(enc_out_dim, embed_dim, kernel_size=1),
                nn.Flatten(2)
            )
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x: dict, batch_positions=None):
        features = {}
        for mod in self.modalities:
            mod_input = {mod: x[mod]}
            if batch_positions is not None:
                out = self.encoders[mod](mod_input, batch_positions=batch_positions)
            else:
                out = self.encoders[mod](mod_input)
                
            if isinstance(out, tuple):
                features[mod] = out[1]
            else:
                features[mod] = out
                
        num_levels = len(features[self.modalities[0]])
        out_features = []
        
        for i in range(num_levels - 1):
            # [B, C_total, H, W]
            level_feat = torch.cat([features[mod][i] for mod in self.modalities], dim=1)
            out_features.append(level_feat)
            
        seqs = []
        spatial_shape = features[self.modalities[0]][-1].shape[-2:]
        
        for mod in self.modalities:
            last_feat = features[mod][-1]
            seq = self.latent_converters[mod](last_feat) 
            seq = seq.transpose(1, 2) 
            seqs.append(seq)
            
        # [B, N_mod * H * W, embed_dim]
        concat_seq = torch.cat(seqs, dim=1)
        
        fused = self.transformer(concat_seq)
        
        B = fused.shape[0]
        mod_spatial = torch.chunk(fused, len(self.modalities), dim=1)
        
        fused_maps = []
        for ms in mod_spatial:
            ms = ms.transpose(1, 2).reshape(B, self.embed_dim, spatial_shape[0], spatial_shape[1])
            fused_maps.append(ms)
            
        final_feat = torch.cat(fused_maps, dim=1)
        out_features.append(final_feat)
        
        return out_features