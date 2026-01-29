import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"
import xarray as xr

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
        """Initialize the PASTIS dataset.

        Args:
            split (str): split of the dataset (train, val).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            fold_config (int): configuration of folds to split the data
        """
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

        self.reference_date = np.datetime64("2019-12-08").astype('datetime64[ns]')
        
        year_start = self.reference_date.astype('datetime64[Y]')
        self.ref_doy = (self.reference_date - year_start).astype('timedelta64[D]').astype(int) + 1

        split_file = os.path.join(self.root_path, f"splits/ssl4eos12_{self.split}.txt")
        with open(split_file, 'r') as f:
            self.samples = f.read().splitlines()

        self.num_classes = 1 # NO LABEL

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.

        Args:
            i (int): index of the item.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {"optical": torch.Tensor},
            "target": torch.Tensor,
            "metadata": 
                {
                "time_linear": torch.Tensor,
                "doy": torch.Tensor,
                "lat": torch.Tensor,
                "len": torch.Tensor
                }
            }.
        """
        file_idx = i // 64
        patch_idx = i % 64

        path = os.path.join(self.root_path, f"{self.split}/{self.modalities[0]}/{self.samples[file_idx][:-4]}")

        with xr.open_zarr(path, consolidated=True) as ds:
            band_data = ds["bands"][patch_idx].transpose("band", "time", "y", "x").values
            lat_val = ds["center_lat"].isel(sample=patch_idx).values
            lon_val = ds["center_lon"].isel(sample=patch_idx).values
            time_val = ds['time_'].isel(sample=patch_idx).values

        optical_ts = torch.from_numpy(band_data).type(torch.float32)
        lat = torch.from_numpy(lat_val).type(torch.float32)
        lon = torch.from_numpy(lon_val).type(torch.float32)

        time_positions = time_val - self.reference_date
        time_positions = torch.from_numpy(time_positions.astype('timedelta64[D]').astype(int)).type(torch.float32)

        if self.multi_temporal == 1:
            # we only take the last frame
            optical_indexes = torch.Tensor([-1]).long()
            optical_ts = optical_ts[:, optical_indexes]

            metadata = torch.Tensor([time_positions[optical_indexes].float()])
        else:
            # select evenly spaced samples
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
            "target": torch.empty(1,1),
            "metadata": {
                "time_linear": metadata,
                "doy": doy_norm,
                "lat": lat_norm,
                "lon": lon_norm
            }
        }

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.samples) * 64

    @staticmethod
    def download():
        pass
