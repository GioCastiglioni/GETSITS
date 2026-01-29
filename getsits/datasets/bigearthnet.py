from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
import geopandas as gpd

from getsits.datasets.base import RawGeoFMDataset


class BigearthNetFull(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        support_test: bool,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: Dict[str, list[str]],
        distribution: list[int],
        data_mean: Dict[str, list[float]],
        data_std: Dict[str, list[float]],
        data_min: Dict[str, list[float]],
        data_max: Dict[str, list[float]],
        download_url: str,
        auto_download: bool,
        fold_config: int,
    ):
        super(BigearthNetFull, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            support_test=support_test,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
            fold_config=fold_config,
        )

        self.root_path = str(root_path)
        self.modalities = ["DATA_S2"]
        self.multi_temporal = int(multi_temporal)

        self.reference_date = np.datetime64("2017-01-01").astype("datetime64[ns]")
        year_start = self.reference_date.astype("datetime64[Y]")
        self.ref_doy = (self.reference_date - year_start).astype("timedelta64[D]").astype(int) + 1

        meta_path = Path(self.root_path) / "metadata_geobench.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata_geobench.parquet not found: {meta_path}")

        gdf = pd.read_parquet(meta_path, engine="pyarrow")

        split_col = "split_geobench"
        #gdf_split = gdf[gdf[split_col] == split].copy()
        if split == "train":
            gdf_split = gdf[gdf[split_col]=="train"].copy()
        elif split == "val":
            gdf_split = gdf[gdf[split_col]=="val"].copy()
        elif split == "test":
            gdf_split = gdf[gdf[split_col]=="test"].copy()

        id_col = "patch_id" 

        gdf_split[id_col] = gdf_split[id_col].astype(str)

        self._meta = gdf_split.set_index(id_col, drop=False)
        self.samples = self._meta[id_col].tolist()


    def _load_s2(self, sample_id: str) -> torch.Tensor:
        s2_path = Path(self.root_path) / "npy_patches" / f"{sample_id}.npy"
        if not s2_path.exists():
            raise FileNotFoundError(f"S2 file missing: {s2_path}")

        arr = np.load(s2_path, allow_pickle=False)

        if arr.ndim == 3:
            C, H, W = arr.shape
            arr = arr[:, None, :, :]

        return torch.from_numpy(arr).float()


    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self._meta.iloc[i]

        sample_id = row["patch_id"]

        date = self._meta.loc[sample_id, "date"]
        optical = self._load_s2(sample_id)
        target = torch.tensor(
            row["labels_encoded"], dtype=torch.float32
        )

        date = torch.tensor(np.datetime64(date).astype("datetime64[D]").astype(int), dtype=torch.float32)
        doy_norm = ((date + self.ref_doy - 1) % 365.25) / 365.25

        lat = float(row["lat"])
        lon = float(row["lon"])

        lat_norm = torch.tensor(lat / 90.0, dtype=torch.float32)
        lon_norm = torch.tensor(lon / 180.0, dtype=torch.float32)

        return {
            "image": {"optical": optical},
            "target": target,
            "metadata": {
                "time_linear": date,
                "doy": torch.tensor([doy_norm], dtype=torch.float32),
                "lat": lat_norm,
                "lon": lon_norm,
            },
        }

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def download():
        pass