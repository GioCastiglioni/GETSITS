import io
import os
import re
import json
import warnings
from itertools import islice

import fsspec
import zarr
import braceexpand
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import IterableDataset

from getsits.datasets.base import RawGeoFMDataset

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"

# --- HELPER FUNCTIONS ---
def extract_modality_names(s):
    pattern = r"\{([^}]*)\}"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []

def multi_tarfile_samples(src_iter):
    for src in src_iter:
        multi_tar_urls = src["url"].translate(str.maketrans("[]", "{}"))
        modality_names = extract_modality_names(multi_tar_urls)
        multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        tar_iters = [wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls]

        try:
            for multi_tar_files in zip(*tar_iters):
                merged_dict = {}
                merged_dict["__key__"] = multi_tar_files[0]["__key__"]
                merged_dict["__url__"] = src["url"]

                for modality_name, modality_dict in zip(modality_names, multi_tar_files):
                    _key = modality_dict.pop("__key__")
                    _url = modality_dict.pop("__url__")
                    for k, v in modality_dict.items():
                        if modality_name is None:
                            merged_dict[k] = v
                        else:
                            merged_dict[f"{modality_name}.{k}"] = v
                yield merged_dict
        except Exception as e:
            #warnings.warn(f"Exception occurred while processing {src['url']}: {repr(e)}. Skipping shard")
            continue

def zarr_metadata_decoding(sample):
    for key, value in list(sample.items()):
        if key.endswith(".zarr.zip"):
            mapper = fsspec.filesystem("zip", fo=io.BytesIO(value), block_size=None).get_mapper("")
            data = zarr.open_consolidated(mapper, mode="r")
            sample[key] = data["bands"][...]
            
            if "center_lon" not in sample.keys():
                sample["center_lon"] = data["center_lon"][...]
                sample["center_lat"] = data["center_lat"][...]
            
            time_key = "time_" + key.split('.')[0]
            sample[time_key] = data["time"][...]
    return sample

def remove_extensions(sample):
    return {os.path.splitext(k.replace(".zip", ""))[0]: v for k, v in sample.items()}


# --- CLASE DATASET ---
class SSL4EOWeb(RawGeoFMDataset, IterableDataset):
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
        distribution: list[float],
        data_mean: dict[str, list[float]],
        data_std: dict[str, list[float]],
        data_min: dict[str, list[float]],
        data_max: dict[str, list[float]],
        download_url: str,
        auto_download: bool,
        fold_config: int
    ):
        safe_root_path = root_path if root_path is not None else "."

        super(SSL4EOWeb, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            support_test=support_test,
            root_path=safe_root_path,
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
        
        self.modalities = ["S2L2A", "S1GRD"] if multi_modal else ["S2L2A"]
        self.reference_date = np.datetime64("2019-12-08").astype('datetime64[ns]')
        year_start = self.reference_date.astype('datetime64[Y]')
        self.ref_doy = (self.reference_date - year_start).astype('timedelta64[D]').astype(int) + 1

        shards = "ssl4eos12_shard_{000001..000477}.tar" if split == "train" else "ssl4eos12_shard_{000001..000005}.tar"
        
        if self.multi_modal:
            urls = f"{self.download_url}/{self.split}/[{','.join(self.modalities)}]/{shards}"
        else:
            urls = f"{self.download_url}/{self.split}/{self.modalities[0]}/{shards}"


        if self.split == "train":
            shard_iterator = wds.ResampledShards(urls, empty_check=False)
            pipeline_steps = [
                shard_iterator,
                wds.split_by_node,
                wds.split_by_worker,
                multi_tarfile_samples if self.multi_modal else wds.tarfile_samples,
                wds.shuffle(1000, initial=100)
            ]
        else:
            shard_iterator = wds.SimpleShardList(urls)
            pipeline_steps = [
                shard_iterator,
                wds.split_by_worker,
                multi_tarfile_samples if self.multi_modal else wds.tarfile_samples,
            ]

        pipeline_steps.extend([
            wds.map(zarr_metadata_decoding),
            wds.map(remove_extensions),
            wds.map(self.format_to_getsits)
        ])

        self.pipeline = wds.DataPipeline(*pipeline_steps)

    def __len__(self) -> int:
        """
        Longitudes exactas basadas en tus logs. 
        Val = 2176 (34 batches * 64). Train = 246144 - 2176 = 243968.
        """
        if self.split == "train":
            total_samples = 243968
        elif self.split == "val" or self.split == "test":
            total_samples = 2176
        else:
            return 0
            
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
            
        return total_samples // world_size

    def __iter__(self):
        # Obtener información del entorno DDP
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        iterator = iter(self.pipeline)
        
        if self.split != "train":
            # VALIDACIÓN: Simulamos el DistributedSampler
            # Ambas GPUs leen los mismos 5 shards, pero se dividen las 
            # imágenes matemáticamente (ej. GPU 0 toma pares, GPU 1 impares).
            for i, sample in enumerate(iterator):
                if i % world_size == rank:
                    yield sample
            return
            
        # ENTRENAMIENTO: El "Cuchillo" de las Épocas
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        samples_per_worker = len(self) // num_workers
        
        for sample in islice(iterator, samples_per_worker):
            yield sample

    def _process_temporal_modality(self, ts_array, time_array):
        ts_tensor = torch.from_numpy(ts_array).float()
        if ts_tensor.shape[0] != ts_tensor.shape[1]: 
            if ts_tensor.shape[1] == len(self.bands["optical"]):
                ts_tensor = ts_tensor.permute(1, 0, 2, 3)

        time_days = (time_array.astype('datetime64[ns]') - self.reference_date).astype('timedelta64[D]').astype(int)
        time_positions = torch.from_numpy(time_days).float()

        if self.multi_temporal == 1:
            idx = torch.tensor([-1]).long()
        else:
            idx = torch.linspace(0, ts_tensor.shape[1] - 1, self.multi_temporal, dtype=torch.long)

        ts_sampled = ts_tensor[:, idx]
        metadata = time_positions[idx]
        doy_norm = ((metadata + self.ref_doy - 1) % 365.25) / 365.25

        meta_dict = {
            "time_linear": metadata,
            "doy": doy_norm
        }
        return ts_sampled, meta_dict

    def format_to_getsits(self, sample):
        image_dict = {}
        meta_dict = {}

        lat_val = float(np.atleast_1d(sample["center_lat"])[0])
        lon_val = float(np.atleast_1d(sample["center_lon"])[0])
        meta_dict["lat"] = torch.tensor(lat_val / 90.0, dtype=torch.float32)
        meta_dict["lon"] = torch.tensor(lon_val / 180.0, dtype=torch.float32)

        if "S2L2A" in sample:
            opt_ts, opt_meta = self._process_temporal_modality(sample["S2L2A"], sample["time_S2L2A"])
            image_dict["optical"] = opt_ts
            meta_dict["optical"] = opt_meta

        if self.multi_modal and "S1GRD" in sample:
            sar_ts, sar_meta = self._process_temporal_modality(sample["S1GRD"], sample["time_S1GRD"])
            image_dict["sar"] = sar_ts
            meta_dict["sar"] = sar_meta

        return {
            "image": image_dict,
            "target": torch.zeros(1),
            "metadata": meta_dict
        }

    @staticmethod
    def download(self) -> None:
        pass