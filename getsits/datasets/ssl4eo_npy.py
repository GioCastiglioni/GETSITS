import os
import json
import numpy as np
import torch
from getsits.datasets.base import RawGeoFMDataset


class SSL4EO(RawGeoFMDataset):
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
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        fold_config: int
    ):
        super(SSL4EO, self).__init__(
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
            fold_config=fold_config
        )
        self.modalities = ["S2L2A"]
        self.nb_split = 1
        self.npy_dir = os.path.join(self.root_path, self.split, "S2L2A_npy")

        self.reference_date = np.datetime64("2019-12-08").astype('datetime64[ns]')
        year_start = self.reference_date.astype('datetime64[Y]')
        self.ref_doy = (self.reference_date - year_start).astype('timedelta64[D]').astype(int) + 1

        # ── Cargar metadata desde GeoJSON ──────────────────────────────────────
        metadata_path = os.path.join(self.root_path, self.split, "S2L2A_metadata.geojson")
        with open(metadata_path, "r") as f:
            geojson = json.load(f)

        # Ordenar por id para garantizar índice consistente con __getitem__
        features = sorted(geojson["features"], key=lambda x: x["properties"]["id"])
        self.metadata_records = [f["properties"] for f in features]
        # ──────────────────────────────────────────────────────────────────────

        self.num_classes = 1  # NO LABEL

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.

        Returns:
            dict with keys:
                "image":    {"optical": Tensor [bands, time, H, W]}
                "target":   Tensor
                "metadata": {"time_linear", "doy", "lat", "lon"}
        """
        record = self.metadata_records[i]

        # ── Leer .npy ─────────────────────────────────────────────────────────
        npy_path = os.path.join(self.npy_dir, record["npy_file"])
        band_data = np.load(npy_path)  # shape real: (time=4, bands=12, H, W)
        band_data = band_data.transpose(1, 0, 2, 3)
        optical_ts = torch.from_numpy(band_data).float()

        # ── Metadata espacial ─────────────────────────────────────────────────
        lat = torch.tensor(record["latitud"], dtype=torch.float32)
        lon = torch.tensor(record["longitud"], dtype=torch.float32)

        # ── Fechas → posiciones temporales ────────────────────────────────────
        # 4 fechas exactas (una por temporada), shape .npy: (4, 12, H, W)
        fechas = [np.datetime64(f, "ns") for f in record["fechas"]]

        time_val = np.array(fechas, dtype="datetime64[ns]")
        time_positions = (time_val - self.reference_date).astype("timedelta64[D]").astype(int)
        time_positions = torch.from_numpy(time_positions).float()

        # ── Selección temporal ────────────────────────────────────────────────
        if self.multi_temporal == 1:
            optical_indexes = torch.tensor([-1]).long()
            optical_ts = optical_ts[:, optical_indexes]
            metadata = torch.tensor([time_positions[optical_indexes].float()])
        else:
            optical_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, self.multi_temporal, dtype=torch.long
            )
            optical_ts = optical_ts[:, optical_indexes]
            metadata = time_positions[optical_indexes].float()

        doy_norm = ((metadata + self.ref_doy - 1) % 365.25) / 365.25
        lat_norm = lat / 90.0
        lon_norm = lon / 180.0

        return {
            "image": {
                "optical": optical_ts,
            },
            "target": torch.empty(1, 1),
            "metadata": {
                "time_linear": metadata,
                "doy": doy_norm,
                "lat": lat_norm,
                "lon": lon_norm,
            },
        }

    def __len__(self) -> int:
        return len(self.metadata_records)

    @staticmethod
    def download():
        pass
