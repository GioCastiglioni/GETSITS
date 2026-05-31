import torch
import torch.nn as nn
from getsits.encoders.base import Encoder

class CoMMEncoder(Encoder):
    def __init__(
        self,
        model_name: str,
        encoders: dict,
        embed_dim: int,
        output_layers: list,
        output_dim: list,
        multi_temporal: bool,
        multi_temporal_output: bool,
        pyramid_output: bool,
        modalities_finetune: dict = None,
        modalities_from_scratch: dict = None,
        encoder_weights: str = "",
        download_url: str = "",
        num_heads: int = 8,
        depth: int = 1,
        positional_encoding: str = "normal",
        input_size: int = 224,
        input_bands: dict = None,
        **kwargs
    ) -> None:
        if input_bands is None:
            input_bands = {}
            for enc in encoders.values():
                if hasattr(enc, 'input_bands') and isinstance(enc.input_bands, dict):
                    input_bands.update(enc.input_bands)
                    
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
        
        self.topology = self.output_dim
        
        self.modalities_finetune = modalities_finetune if modalities_finetune is not None else {}
        self.modalities_from_scratch = modalities_from_scratch if modalities_from_scratch is not None else {}

        self.encoders = nn.ModuleDict(encoders)
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

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.topology[-1], 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, 512)
        )
        
        self._freeze_modalities()

    def _freeze_modalities(self):
        for mod in self.modalities:
            if not self.modalities_finetune.get(mod, True):
                for param in self.encoders[mod].parameters():
                    param.requires_grad = False
                for param in self.latent_converters[mod].parameters():
                    param.requires_grad = False

    def load_encoder_weights(self, logger, from_scratch=False):
        if not from_scratch:
            if self.encoder_weights:
                logger.info(f"Loading full pre-trained CoMMEncoder weights from {self.encoder_weights}...")
                checkpoint = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
                
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
                
                self.load_state_dict(state_dict, strict=False)
                logger.info("Pre-trained CoMMEncoder weights loaded successfully.")
            else:
                logger.warning("Global 'from_scratch' is False but 'encoder_weights' is missing.")
        else:
            for mod in self.modalities:
                mod_from_scratch = self.modalities_from_scratch.get(mod, True)
                if hasattr(self.encoders[mod], 'load_encoder_weights'):
                    self.encoders[mod].load_encoder_weights(logger, from_scratch=mod_from_scratch)
                else:
                    logger.info(f"Modality '{mod}' does not support weight loading via load_encoder_weights.")
        
    def forward(self, x: dict | torch.Tensor, batch_positions=None, projection=False):
        if projection:
            return self.projector(x)

        features = {}
        temporal_reduced = {}
        
        for mod in self.modalities:
            mod_input = {mod: x[mod]}
            if batch_positions is not None:
                out_tuple = self.encoders[mod](mod_input, batch_positions=batch_positions)
            else:
                out_tuple = self.encoders[mod](mod_input)
                
            if isinstance(out_tuple, tuple):
                temporal_reduced[mod] = out_tuple[0]  
                features[mod] = out_tuple[1]         
            else:
                features[mod] = out_tuple
                temporal_reduced[mod] = out_tuple[-1]
                
        num_levels = len(features[self.modalities[0]])
        out_features = []
        
        for i in range(num_levels - 1):
            dim_c = 2 if features[self.modalities[0]][i].dim() == 5 else 1
            level_feat = torch.cat([features[mod][i] for mod in self.modalities], dim=dim_c)
            out_features.append(level_feat)
            
        seqs = []
        spatial_shape = temporal_reduced[self.modalities[0]].shape[-2:]
        
        for mod in self.modalities:
            last_feat = temporal_reduced[mod] 
            seq = self.latent_converters[mod](last_feat) 
            seq = seq.transpose(1, 2) 
            seqs.append(seq)
            
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