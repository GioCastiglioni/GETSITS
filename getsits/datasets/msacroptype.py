from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import geopandas as gpd

from getsits.datasets.base import RawGeoFMDataset


class SouthAfricaCrops(RawGeoFMDataset):
    """
    Works with your processed dataset structure:
    root_path/
      metadata_formatted.geojson
      DATA_S2/S2_<ID>.npy
      ANNOTATIONS/ParcelIDs_<ID>.npy
    """

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
        super(SouthAfricaCrops, self).__init__(
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

        if self.split == "val":
            self.split = "valid"

        self.root_path = str(root_path)
        self.modalities = ["DATA_S2"]
        self.multi_temporal = int(multi_temporal)

        self.reference_date = np.datetime64("2017-03-01").astype("datetime64[ns]")
        year_start = self.reference_date.astype("datetime64[Y]")
        self.ref_doy = (self.reference_date - year_start).astype("timedelta64[D]").astype(int) + 1

        meta_path = Path(self.root_path) / "metadata_formatted.geojson"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata_formatted.geojson not found: {meta_path}")

        gdf = gpd.read_file(meta_path)

        split_col = "split_geobench"


        gdf_split = gdf[gdf[split_col] == self.split].copy()

        id_col = "ID_PATCH" 

        gdf_split[id_col] = gdf_split[id_col].astype(str)

        self._meta = gdf_split.set_index(id_col, drop=False)
        self.samples = self._meta[id_col].tolist()

        self.has_latlon = ("lat" in self._meta.columns) and ("lon" in self._meta.columns)

    def _load_s2(self, sample_id: str) -> torch.Tensor:
        s2_path = Path(self.root_path) / "DATA_S2" / f"S2_{sample_id}.npy"
        if not s2_path.exists():
            raise FileNotFoundError(f"S2 file missing: {s2_path}")

        arr = np.load(s2_path, allow_pickle=False)

        if arr.ndim == 3:
            C, H, W = arr.shape
            arr = arr[:, None, :, :]

        return torch.from_numpy(arr).float()

    def _load_ann(self, sample_id: str) -> torch.Tensor:
        ann_path = Path(self.root_path) / "ANNOTATIONS" / f"ParcelIDs_{sample_id}.npy"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file missing: {ann_path}")

        y = np.load(ann_path, allow_pickle=False)
        return torch.from_numpy(y).long()

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample_id = self.samples[i]

        optical = self._load_s2(sample_id)
        target = self._load_ann(sample_id)

        date = self._meta.loc[sample_id, "date"]
        ref_days_int = self.reference_date.astype("datetime64[D]").astype(int)
        date = np.datetime64(date).astype("datetime64[D]").astype(int)
        doy = (date + self.ref_doy - 1) % 365.25

        doy_norm = torch.tensor([doy], dtype=torch.float32) / 365.25
        
        days_to_ref = torch.tensor([date - ref_days_int], dtype=torch.float32)

        lat = float(self._meta.loc[sample_id, "lat"])
        lon = float(self._meta.loc[sample_id, "lon"])
        lat_norm = torch.tensor(lat / 90.0, dtype=torch.float32)
        lon_norm = torch.tensor(lon / 180.0, dtype=torch.float32)

        return {
            "image": {"optical": optical[:-1]},
            "target": target,
            "metadata": {
                "time_linear": days_to_ref,
                "doy": doy_norm,
                "lat": lat_norm,
                "lon": lon_norm,
            },
        }

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def download():
        pass