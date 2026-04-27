import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import torch
local_rank = 0
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader
torch.backends.cuda.preferred_linalg_library("magma")
import gc
import numpy as np

from getsits.datasets.base import GeoFMDataset, RawGeoFMDataset
from getsits.encoders.base import Encoder
from getsits.utils.collate_fn import get_collate_fn
from getsits.utils.logger import init_logger
from getsits.utils.utils import (
    seed_worker,
)

args = """
dataset=ssl4eov1_1
task=pretraining
encoder=vit_small
encoder.positional_encoding="geotime"
batch_size=12
test_batch_size=12
preprocessing=pretrain_default
criterion=lejepa
"""

overrides_list = args.split()

GlobalHydra.instance().clear()

with initialize(version_base=None, config_path="configs"):
    
    cfg = compose(
        config_name="pretrain", 
        overrides=overrides_list 
    )

def calculate_jepa_score_batch(model, batch, current_scores_list, eps=1e-8):
    model.eval() 
    
    images = batch["image"]["optical"] 
    current_metadata_batch = batch["metadata"]
    
    batch_size = images.shape[0]
    device = images.device

    img_batch = images.clone().detach().requires_grad_(True)
    
    with torch.enable_grad():
        output_full = model(img_batch, current_metadata_batch)
        embedding_batch = output_full.mean(dim=(-2, -1))
        embedding_batch = embedding_batch.reshape(batch_size, -1)
    
    n_features = embedding_batch.shape[1] 
    
    jacobian_slices_cpu = []
    
    for j in range(n_features):
        grad_output = torch.zeros_like(embedding_batch)
        grad_output[:, j] = 1.0
        retain_graph = (j < n_features - 1)
        
        grads = torch.autograd.grad(
            outputs=embedding_batch,
            inputs=img_batch,
            grad_outputs=grad_output,
            retain_graph=retain_graph,
            create_graph=False
        )[0]
        
        grads_flat = grads.detach().cpu().reshape(batch_size, -1)
        jacobian_slices_cpu.append(grads_flat)
        
        del grads, grad_output, grads_flat

    J_batch_cpu = torch.stack(jacobian_slices_cpu).permute(1, 0, 2).float()
    
    del output_full, embedding_batch, img_batch
    torch.cuda.empty_cache()
    
    try:
        svdvals_batch = torch.linalg.svdvals(J_batch_cpu)
        scores_batch = svdvals_batch.clip(min=eps).log().sum(dim=1)
        current_scores_list.extend(scores_batch.tolist())
        
    except RuntimeError as e:
        print(f"Error SVD Batch: {e}")
        current_scores_list.extend([float('nan')] * batch_size)
            
    del J_batch_cpu, svdvals_batch
    gc.collect()

    return current_scores_list


logger = init_logger("./logger.txt", rank=0)

encoder: Encoder = instantiate(cfg.encoder)
encoder.load_encoder_weights(logger, from_scratch=cfg.from_scratch)

class Wrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder
    
    def forward(self, img, batch_positions):
        _, feat, _, att = self.encoder(img, batch_positions)
        
        return self.collapse_T(feat, att)[-1]

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

model = Wrapper(encoder).to(device)
model.eval()

modalities = list(encoder.input_bands.keys())
collate_fn = get_collate_fn(modalities)

test_preprocessor = instantiate(
                cfg.preprocessing.val,
                dataset_cfg=cfg.dataset,
                encoder_cfg=cfg.encoder,
                _recursive_=False,
            )
raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")
test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor, cfg.data_replicate)

test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False, #causes memory leak
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn,
        )

scores = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch["image"]["optical"] = batch["image"]["optical"].to(device).requires_grad_(True)
        batch["metadata"] = {k: v.to(device) for k, v in batch["metadata"].items()}
        
        scores = calculate_jepa_score_batch(
            model, batch, scores
        )
        
        del batch
        torch.cuda.empty_cache()
        np.save("./jepa_scores/ssl4eo.npy", scores)