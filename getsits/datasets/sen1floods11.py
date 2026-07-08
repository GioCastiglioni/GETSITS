import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
import torch

from datetime import datetime

from getsits.datasets.base import RawGeoFMDataset
from getsits.datasets.utils import download_bucket_concurrently


class Sen1Floods11(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        gcs_bucket: str,
        support_test: bool,
        fold_config: int,
    ):
        self.gcs_bucket = gcs_bucket

        super(Sen1Floods11, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
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
            support_test=support_test
        )

        self.root_path = root_path
        self.classes = classes
        self.split = split

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download

        self.split_mapping = {"train": "train", "val": "valid", "test": "test"}

        split_file = os.path.join(
            self.root_path,
            "v1.1",
            f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv",
        )
        metadata_file = os.path.join(
            self.root_path, "v1.1", "Sen1Floods11_Metadata.geojson"
        )
        data_root = os.path.join(
            self.root_path, "v1.1", "data/flood_events/HandLabeled/"
        )

        self.metadata = geopandas.read_file(metadata_file)
        self.metadata[['approx_lat', 'approx_lon']] = self.metadata.apply(self.get_centroid, axis=1)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) 
            for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[1]) for f in file_list
        ]

    def __len__(self):
        return len(self.s2_image_list)

    def get_centroid(self, row):
        coords = row['geometry'].centroid
        return pd.Series([coords.y, coords.x])

    def _get_metadata(self, index, reference_date_str="2016-08-12"):
        reference_date = pd.to_datetime(reference_date_str, dayfirst=True)
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        location = "Cambodia" if location == "Mekong" else location
        
        # Extraer la fila correspondiente a la ubicación
        row = self.metadata[self.metadata["location"] == location].iloc[0]

        # Extraer fechas independientes
        s2_date = pd.to_datetime(row["s2_date"])
        # Si s1_date está en el GeoJSON lo usamos, si no, usamos un fallback seguro
        s1_date = pd.to_datetime(row["s1_date"])

        s2_doy = s2_date.timetuple().tm_yday
        s1_doy = s1_date.timetuple().tm_yday
        
        lat = row["approx_lat"]
        lon = row["approx_lon"]
        
        s2_delta_days = (s2_date - reference_date).days
        s1_delta_days = (s1_date - reference_date).days
        
        return s2_delta_days, s2_doy, s1_delta_days, s1_doy, lat, lon

    def __getitem__(self, index):
        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)

        s2_ts, s2_doy, s1_ts, s1_doy, lat, lon = self._get_metadata(index)

        lat_norm = torch.tensor(lat, dtype=torch.float32) / 90.0
        lon_norm = torch.tensor(lon, dtype=torch.float32) / 180.0

        # [C, 1, H, W]
        s2_image = torch.from_numpy(s2_image).float().unsqueeze(1)
        s1_image = torch.from_numpy(s1_image).float().unsqueeze(1)
        target = torch.from_numpy(target).long()

        output = {
            "image": {
                "optical": s2_image,
                "sar": s1_image,
            },
            "target": target,
            "metadata": {
                "optical": {
                    "time_linear": torch.tensor([s2_ts], dtype=torch.float32),
                    "doy": torch.tensor([s2_doy], dtype=torch.float32) / 365.25,
                    "lat": lat_norm,
                    "lon": lon_norm
                },
                "sar": {
                    "time_linear": torch.tensor([s1_ts], dtype=torch.float32),
                    "doy": torch.tensor([s1_doy], dtype=torch.float32) / 365.25,
                    "lat": lat_norm,
                    "lon": lon_norm
                },
            }
        }

        return output

    @staticmethod
    def download(self, silent=False):
        if os.path.exists(self.root_path):
            if not silent:
                print(
                    "Sen1Floods11 Dataset folder exists, skipping downloading dataset."
                )
            return
        download_bucket_concurrently(self.gcs_bucket, self.root_path)