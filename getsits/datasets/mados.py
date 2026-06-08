import os
import time 
import pathlib
import urllib.request
import urllib.error
import zipfile

from glob import glob
import rasterio
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from getsits.datasets.base import RawGeoFMDataset

class MADOS(RawGeoFMDataset):
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
        """Initialize the MADOS dataset.
        Link: https://marine-pollution.github.io/index.html
        """
        super(MADOS, self).__init__(
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

        self.ROIs_split = np.genfromtxt(os.path.join(self.root_path, 'splits', f'{split}_X.txt'), dtype='str')

        self.image_list = []
        self.target_list = []

        self.tiles = sorted(glob(os.path.join(self.root_path, '*')))

        for tile in self.tiles:
            splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]

            for crop in splits:
                crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]

                if crop_name in self.ROIs_split:
                    all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_' + crop))
                    all_bands = sorted(all_bands, key=self.get_band)

                    self.image_list.append(all_bands)

                    cl_path = os.path.join(tile, '10', os.path.basename(tile) + '_L2R_cl_' + crop)
                    self.target_list.append(cl_path)

    def __len__(self):
        return len(self.image_list)

    def getnames(self):
        return self.ROIs_split

    def __getitem__(self, index):
        all_bands = self.image_list[index]
        current_image = []
        for c, band in enumerate(all_bands):
            upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
            with rasterio.open(band, mode='r') as src:
                this_band = src.read(1,
                                     out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)),
                                     resampling=rasterio.enums.Resampling.nearest
                                     )
                this_band = torch.from_numpy(this_band)
                current_image.append(this_band)

        image = torch.stack(current_image)
        invalid_mask = torch.isnan(image)
        image[invalid_mask] = 0

        # images must be of shape (C, T, H, W).
        image = image.unsqueeze(1).to(torch.float32)

        with rasterio.open(self.target_list[index], mode='r') as src:
            target = src.read(1)
        
        target = torch.from_numpy(target.astype(np.int64))
        target = target - 1

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {
                "time_linear": torch.tensor([0.0], dtype=torch.float32),
                "doy": torch.tensor([0.0], dtype=torch.float32),
                "lat": torch.tensor(0.0, dtype=torch.float32),
                "lon": torch.tensor(0.0, dtype=torch.float32)
            }
        }

        return output

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

    @staticmethod
    def download(self, silent=False):
        output_path = pathlib.Path(self.root_path)
        url = self.download_url

        existing_dirs = list(output_path.glob("Scene_*"))
        if existing_dirs:
            if not silent:
                print("MADOS Dataset folder exists, skipping downloading dataset.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        temp_file_name = f"temp_{hex(int(time.time()))}_MADOS.zip"
        
        try:
            from tqdm import tqdm
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
            
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, output_path / temp_file_name, reporthook=t.update_to)
        except ImportError:
            urllib.request.urlretrieve(url, output_path / temp_file_name)
        except urllib.error.HTTPError as e:
            print('Error while downloading dataset: The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            return
        except urllib.error.URLError as e:
            print('Error while downloading dataset: Failed to reach a server.')
            print('Reason: ', e.reason)
            return

        with zipfile.ZipFile(output_path / temp_file_name, 'r') as zip_ref:
            print(f"Extracting to {output_path} ...")
            members = []
            for zipinfo in zip_ref.infolist():
                new_path = os.path.join(*(zipinfo.filename.split(os.path.sep)[1:]))
                zipinfo.filename = str(new_path)
                members.append(zipinfo)

            zip_ref.extractall(output_path, members)
            print("done.")

        (output_path / temp_file_name).unlink()