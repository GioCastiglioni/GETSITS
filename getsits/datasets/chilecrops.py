import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
from einops import rearrange
import geopandas as gpd
import re

from getsits.datasets.base import RawGeoFMDataset


def temporal_subsampling(desired_length: int, curr: torch.Tensor, subsets: list = [1, 6, 15, 25, 35]):
    assert desired_length in subsets, "desired_length must be in the list of allowed subsets"
    subsets = subsets.copy()
    k = subsets.pop()
    curr = curr[torch.linspace(0, curr.shape[0] - 1, k, dtype=torch.long)]
    if desired_length == k:
        return curr
    else:
        return temporal_subsampling(desired_length, curr, subsets)


class CropChileDataset(RawGeoFMDataset):

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
        bands: dict,
        distribution: list,
        data_mean: dict,
        data_std: dict,
        data_min: dict,
        data_max: dict,
        download_url: str,
        auto_download: bool,
        fold_config: int,
        label_type: str = "macro",
        reference_date: str = "2016-01-01",
    ):
        super(CropChileDataset, self).__init__(
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

        folds_dict = {
            1: {"train": [1, 2, 3], "val": [4], "test": [5]},
            2: {"train": [2, 3, 4], "val": [5], "test": [1]},
            3: {"train": [3, 4, 5], "val": [1], "test": [2]},
            4: {"train": [4, 5, 1], "val": [2], "test": [3]},
            5: {"train": [5, 1, 2], "val": [3], "test": [4]},
        }

        assert split in ["train", "val", "test"], \
            "Split must be 'train', 'val' or 'test'"

        target_folds = folds_dict[fold_config][split]

        self.label_type = label_type
        self.modalities = ["S2", "elevation", "landform", "mTPI"]

        # ── Fecha de referencia ───────────────────────────────
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.ref_doy = self.reference_date.timetuple().tm_yday

        # ── Directorios ───────────────────────────────────────
        root = Path(root_path)
        self.data_dir = root / "data"
        self.labels_dir = root / "labels"
        self.labels_macro_dir = root / "labels_macro"
        self.elevation_dir = root / "elevation"
        self.landform_dir = root / "landform"
        self.mtpi_dir = root / "mTPI"

        # ── metadata_final.geojson ────────────────────────────
        geojson_path = root / "metadata" / "metadata_final.geojson"
        if not geojson_path.exists():
            raise FileNotFoundError(f"No se encontró metadata_final.geojson en {root / 'metadata'}")

        gdf = gpd.read_file(geojson_path)

        # Parsear dates_s2 si vienen como string
        def parse_dates(val):
            if val is None:
                return []
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], str) and len(val[0]) == 10:
                    return val
                if len(val) == 1 and isinstance(val[0], str):
                    val = val[0]
            if isinstance(val, str):
                return re.findall(r'\d{4}-\d{2}-\d{2}', val)
            return []

        gdf["dates_s2"] = gdf["dates_s2"].apply(parse_dates)

        # Construir DataFrame de metadata
        records = []
        for _, row in gdf.iterrows():
            records.append({
                "id": int(row["id"]),
                "tile": row["tile"],
                "patch_id": int(row["patch_id"]),
                "crop_type": row["crop_type"],
                "split": int(row["split"]),
                "dates": row["dates_s2"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            })

        self.meta_patch = pd.DataFrame(records)
        self.meta_patch.index = range(len(self.meta_patch))

        # ── Filtrar por fold/split ────────────────────────────
        self.meta_patch = self.meta_patch[
            self.meta_patch["split"].isin(target_folds)
        ].copy()
        self.meta_patch.index = range(len(self.meta_patch))

        print(f"[{split.upper()} | fold {fold_config}] "
              f"{len(self.meta_patch)} patches | folds {target_folds}")

        # ── Tabla de fechas relativas ─────────────────────────
        self.date_range = np.arange(-500, 10000, dtype=int)
        self.date_tables = {mod: None for mod in self.modalities}

        date_table = pd.DataFrame(
            0, index=self.meta_patch.index,
            columns=self.date_range, dtype=int
        )
        for pid, row in self.meta_patch.iterrows():
            date_list = row.get("dates", [])
            if not isinstance(date_list, (list, tuple, np.ndarray)):
                continue
            rel = pd.Series(date_list).apply(
                lambda x: (datetime.strptime(x, "%Y-%m-%d") - self.reference_date).days
            )
            rel = rel[
                (rel >= self.date_range.min()) & (rel <= self.date_range.max())
            ]
            if len(rel) > 0:
                date_table.loc[pid, rel.values] = 1

        self.date_tables["S2"] = {
            idx: date_table.loc[idx].to_numpy(dtype=int)
            for idx in date_table.index
        }

    # ── Helpers de carga ──────────────────────────────────────────────────────

    def _get_base_name(self, row: pd.Series) -> str:
        """Genera el nombre base: {tile}_{id:05d}"""
        return f"{row['tile']}_{int(row['id']):05d}"

    def _load_s2(self, base_name: str) -> Optional[np.ndarray]:
        """data/S2_{base_name}.npy → (T, C, H, W)"""
        path = self.data_dir / f"S2_{base_name}.npy"
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def _load_label(self, base_name: str) -> Optional[np.ndarray]:
        """labels/label_{base_name}.npy o labels_macro/macro_{base_name}.npy → (H, W)"""
        if self.label_type == "macro":
            path = self.labels_macro_dir / f"macro_{base_name}.npy"
        else:
            path = self.labels_dir / f"label_{base_name}.npy"
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def _load_static(self, directory: Path, prefix: str, base_name: str) -> Optional[np.ndarray]:
        """{dir}/{prefix}_{base_name}.npy → (1, H, W)"""
        path = directory / f"{prefix}_{base_name}.npy"
        if not path.exists():
            return None
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return arr

    # ── Fechas ────────────────────────────────────────────────────────────────

    def get_dates(self, id_patch: int, sat: str = "S2") -> torch.Tensor:
        table = self.date_tables.get(sat)
        if table is None:
            return torch.empty(0, dtype=torch.int32)
        row = table.get(id_patch)
        if row is None:
            return torch.empty(0, dtype=torch.int32)
        indices = np.where(row == 1)[0]
        if indices.size == 0:
            return torch.empty(0, dtype=torch.int32)
        rel_days = self.date_range[indices]
        return torch.tensor(rel_days, dtype=torch.int32)

    # ── __len__ / __getitem__ ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.meta_patch)

    def __getitem__(self, i: int) -> dict:
        row = self.meta_patch.iloc[i]
        id_patch = self.meta_patch.index[i]
        base_name = self._get_base_name(row)

        # ── lat / lon ─────────────────────────────────────────
        lat = float(row["lat"])
        lon = float(row["lon"])
        lat_norm = torch.tensor(lat / 90.0, dtype=torch.float32)
        lon_norm = torch.tensor(lon / 180.0, dtype=torch.float32)

        # ── S2: (C, T, H, W) ──────────────────────────────────
        s2 = self._load_s2(base_name)
        if s2 is None:
            raise FileNotFoundError(f"S2 no encontrado: S2_{base_name}.npy")

        s2 = s2.astype(np.float32)
        if s2.ndim == 3:
            s2 = s2[np.newaxis]

        s2_tensor = torch.from_numpy(s2)
        optical_ts = rearrange(s2_tensor, "t c h w -> c t h w")

        optical_indexes = torch.linspace(
            0, optical_ts.shape[1] - 1, self.multi_temporal, dtype=torch.long
        )
        optical_ts = optical_ts[:, optical_indexes]

        # ── Fechas S2 ─────────────────────────────────────────
        dates = self.get_dates(id_patch, "S2")
        if dates.numel() >= self.multi_temporal:
            d_idx = torch.linspace(
                0, dates.shape[0] - 1, self.multi_temporal, dtype=torch.long
            )
            metadata = dates[d_idx].float()
        elif dates.numel() > 0:
            pad_size = self.multi_temporal - dates.numel()
            metadata = torch.cat([dates.float(), dates[-1].float().repeat(pad_size)])
        else:
            metadata = torch.zeros(self.multi_temporal, dtype=torch.float32)

        metadata = metadata[:self.multi_temporal]
        doy_norm = ((metadata + self.ref_doy - 1) % 365) / 365

        # ── Estáticos: (1, 1, H, W) ───────────────────────────
        static_cfg = {
            "elevation": (self.elevation_dir, "elevation"),
            "landform": (self.landform_dir, "landform"),
            "mTPI": (self.mtpi_dir, "mTPI"),
        }

        image = {"optical": optical_ts.to(torch.float32)}

        sar_channels = []
        for mod, (directory, prefix) in static_cfg.items():
            arr = self._load_static(directory, prefix, base_name)
            if arr is None:
                sar_channels.append(torch.zeros(1, 1, self.img_size, self.img_size))
            else:
                sar_channels.append(torch.from_numpy(arr).unsqueeze(1))

        image["sar"] = torch.cat(sar_channels, dim=0).to(torch.float32)

        label = self._load_label(base_name)
        if label is not None:
            target = torch.from_numpy(label.astype(np.int64))
        else:
            target = torch.zeros(self.img_size, self.img_size, dtype=torch.int64)

        return {
            "image": image,
            "target": target,
            "metadata": {
                "optical": {
                    "time_linear": metadata,
                    "doy": doy_norm,
                },
                "sar": {
                    "time_linear": torch.zeros(1, dtype=torch.float32),
                    "doy": torch.zeros(1, dtype=torch.float32),
                },
                "lat": lat_norm,
                "lon": lon_norm,
            },
        }

    @staticmethod
    def download():
        pass