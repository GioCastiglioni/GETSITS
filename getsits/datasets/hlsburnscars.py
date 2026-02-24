import os
import pathlib
import tarfile
import time
import urllib
from glob import glob
from typing import Sequence, Tuple
from datetime import datetime

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split

from getsits.datasets.base import RawGeoFMDataset
from getsits.datasets.utils import DownloadProgressBar


class HLSBurnScars(RawGeoFMDataset):
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
        support_test: bool,
        fold_config: int,
    ):
        """Initialize the HLSBurnScars dataset.
        Link: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars

        Args:
            split (str): split of the dataset (train, val, test).
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
        """

        super(HLSBurnScars, self).__init__(
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
            support_test=support_test,
            fold_config=fold_config,
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

        reference_date = "2018-01-01"
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.split_mapping = {
            "train": "training",
            "val": "training",
            "test": "validation",
        }

        all_files = sorted(
            glob(
                os.path.join(
                    self.root_path, self.split_mapping[self.split], "*merged.tif"
                )
            )
        )
        all_targets = sorted(
            glob(
                os.path.join(
                    self.root_path, self.split_mapping[self.split], "*mask.tif"
                )
            )
        )

        if self.split != "test":
            split_indices = self.get_train_val_split(all_files)
            if self.split == "train":
                indices = split_indices["train"]
            else:
                indices = split_indices["val"]
            self.image_list = [all_files[i] for i in indices]
            self.target_list = [all_targets[i] for i in indices]
        else:
            self.image_list = all_files
            self.target_list = all_targets

    @staticmethod
    def get_train_val_split(all_files) -> Tuple[Sequence[int], Sequence[int]]:
        # Fixed stratified sample to split data into train/val.
        # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set.
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=23,
        )
        return {"train": train_idxs, "val": val_idxs}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        file = self.image_list[index]
        filename = file.split("/")[-1]
        image = tiff.imread(file)
        image = image.astype(np.float32)  # Convert to float32
        image = torch.from_numpy(image).permute(2, 0, 1)

        target = tiff.imread(self.target_list[index])
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        invalid_mask = image == 9999
        image[invalid_mask] = 0

        meta_data = filename.split('.')
        tile_id = meta_data[2]
        date_str = meta_data[3]
        
        date_obj = datetime.strptime(date_str, "%Y%j")
        
        days_diff = (date_obj - self.reference_date).days
        time_linear = torch.tensor([days_diff], dtype=torch.float32)
        
        doy = date_obj.timetuple().tm_yday
        doy_norm = torch.tensor([(doy - 1) / 365.25], dtype=torch.float32)
        
        lat, lon = self.get_tile_centroid(tile_id)
        
        lat_norm = torch.tensor(lat / 90.0, dtype=torch.float32)
        lon_norm = torch.tensor(lon / 180.0, dtype=torch.float32)
        
        image = image.unsqueeze(1)
        output = {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {
                "time_linear": time_linear,
                "doy": doy_norm,
                "lat": lat_norm,
                "lon": lon_norm
            }
        }

        return output

    @staticmethod
    def download(self, silent=False):
        output_path = pathlib.Path(self.root_path)
        url = self.download_url

        try:
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            if not silent:
                print(
                    "HLSBurnScars dataset folder exists, skipping downloading dataset."
                )
            return

        temp_file_name = f"temp_{hex(int(time.time()))}_hls_burn_scars.tar.gz"
        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(url, output_path / temp_file_name, pbar)
        except urllib.error.HTTPError as e:
            print(
                "Error while downloading dataset: The server couldn't fulfill the request."
            )
            print("Error code: ", e.code)
            return
        except urllib.error.URLError as e:
            print("Error while downloading dataset: Failed to reach a server.")
            print("Reason: ", e.reason)
            return

        with tarfile.open(output_path / temp_file_name, "r") as tar:
            print(f"Extracting to {output_path} ...")
            tar.extractall(output_path)
            print("done.")

        os.remove(output_path / temp_file_name)

