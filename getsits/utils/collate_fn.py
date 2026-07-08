from typing import Callable

import torch
import torch.nn.functional as F


def get_collate_fn(modalities: list[str]) -> Callable:
    def collate_fn(
        batch: list[dict[str, dict[str, torch.Tensor] | torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Collate function for torch DataLoader
        
        Args:
            batch: list of dictionaries with keys 'image', 'target', and optionally 'metadata'.
                   'image' is a dict with keys corresponding to modalities and values being torch.Tensor.
                   For single images: shape (C, H, W), for time series: (C, T, H, W).
                   'target' is a torch.Tensor.
                   'metadata' is a dictionary or torch.Tensor if present.
        Returns:
            A dictionary with keys 'image', 'target', and 'metadata' (if available).
        """
        T_max = 0
        for modality in modalities:
            for x in batch:
                if len(x["image"][modality].shape) == 4:
                    T_max = max(T_max, x["image"][modality].shape[1])
                    
        for modality in modalities:
            for i, x in enumerate(batch):
                if len(x["image"][modality].shape) == 4:
                    T = x["image"][modality].shape[1]
                    if T < T_max:
                        padding = (0, 0, 0, 0, 0, T_max - T)
                        batch[i]["image"][modality] = F.pad(
                            x["image"][modality], padding, "constant", 0
                        )

        batch_out = {
            "image": {
                modality: torch.stack([x["image"][modality] for x in batch])
                for modality in modalities
            },
            "target": torch.stack([x["target"] for x in batch]),
        }
        
        def stack_metadata(metadata_list: list) -> dict:
            """Recursively stacks nested metadata dictionaries."""
            result = {}
            ref_item = metadata_list[0]
            for key in ref_item.keys():
                if isinstance(ref_item[key], dict):
                    result[key] = stack_metadata([item[key] for item in metadata_list])
                elif isinstance(ref_item[key], torch.Tensor):
                    result[key] = torch.stack([item[key] for item in metadata_list])
                else:
                    try:
                        result[key] = torch.tensor([item[key] for item in metadata_list])
                    except Exception:
                        result[key] = [item[key] for item in metadata_list]
            return result

        if "metadata" in batch[0]:
            meta_example = batch[0]["metadata"]

            if isinstance(meta_example, dict):
                batch_out["metadata"] = stack_metadata([item["metadata"] for item in batch])
            elif isinstance(meta_example, torch.Tensor):
                batch_out["metadata"] = torch.stack([item["metadata"] for item in batch])
        else:
            ref_modality = modalities[0]
            T = batch_out["image"][ref_modality].shape[2]
            B = len(batch)
            batch_out["metadata"] = torch.stack([torch.linspace(0, 999, T).long() for _ in range(B)])
            
        return batch_out

    return collate_fn