import numpy as np
import torch
import os
from getsits.datasets.base import RawGeoFMDataset
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from getsits.datasets.utils import decompress_zip_with_progress
try:
    import geobench
except ImportError:
    print("geobench not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "geobench"])
    import geobench


class mNzCattle(RawGeoFMDataset):
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
        super(mNzCattle, self).__init__(
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

        self.root_path = root_path
        self.split = split
        
        split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
        
        task = geobench.load_task_specs(self.root_path)
        self.dataset = task.get_dataset(split=split_mapping[self.split])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        all_band_names = ("Blue", "Green", "Red")
        rgb_bands = ("Red", "Green", "Blue")
        BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
            
        image, band_names = sample.pack_to_3d(band_names=BAND_SETS["rgb"])
        label = sample.label.data
        filename = sample.sample_name
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() 
        image=image.unsqueeze(1)

        return {
            "image": {
                "optical": image,
            },
            "target": torch.tensor(label, dtype=torch.int64),
            "filename": filename,
            "metadata": {
                "time_linear": torch.tensor([0.0], dtype=torch.float32),
                "doy": torch.tensor([0.0], dtype=torch.float32),
                "lat": torch.tensor(0.0, dtype=torch.float32),
                "lon": torch.tensor(0.0, dtype=torch.float32)
            },
        }
    
    def download(self, silent=False):
        local_directory = Path(os.getenv("GEO_BENCH_DIR"))
        dataset_repo = self.download_url

        local_directory.mkdir(parents=True, exist_ok=True)

        api = HfApi()
        dataset_files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")

        for file in dataset_files:
            if file not in ['segmentation_v1.0/m-nz-cattle.zip', 'segmentation_v1.0/normalizer.json']:
                continue

            local_file_path = local_directory / file
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=dataset_repo,
                filename=file,
                cache_dir=local_directory,
                local_dir=local_directory,
                repo_type="dataset",
            )
            if file.endswith(".zip"):
                print(f"Decompressing ...")
                decompress_zip_with_progress(local_directory / file)