import logging
import operator
import os
import pathlib
import time

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from getsits.utils.logger import RunningAverageMeter, sec_to_hm
from getsits.utils.utils import LeJEPATransform as ConsistentTransform
from torch.distributed.nn import all_reduce as functional_all_reduce

def all_reduce(x, op="AVG"):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        op_enum = torch.distributed.ReduceOp.AVG if op.upper() == "AVG" else torch.distributed.ReduceOp.SUM
        return functional_all_reduce(x, op=op_enum)
    else:
        return x
    
def dict_to_device(d, device):
    """Lleva diccionarios anidados recursivamente al dispositivo."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            res[k] = v.to(device)
        else:
            res[k] = v
    return res

def slice_metadata(d, indices, T):
    """Recorta la dimensión temporal de los metadatos de forma recursiva."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = slice_metadata(v, indices, T)
        elif isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == T:
            res[k] = v[:, indices]
        else:
            res[k] = v
    return res

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        n_global: int = 2,
        n_local: int = 8
    ):
        """Initialize the Trainer.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_per_epoch = len(self.train_loader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval

        self.training_stats = {
            name: RunningAverageMeter(length=self.batch_per_epoch)
            for name in ["loss", "data_time", "batch_time"]
        }
        self.training_metrics = {}
        self.best_metric = float("inf")
        self.best_metric_comp = operator.lt
        self.num_classes = self.train_loader.dataset.num_classes

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enable_mixed_precision)

        self.start_epoch = 0

        if self.use_wandb:
            import wandb
            self.wandb = wandb
        
        self.transform = ConsistentTransform(h_w=self.model.module.input_size, degrees=45).to(self.device)

        self.n_global = n_global
        self.n_local = n_local
        
        if not hasattr(self.criterion, "global_step_views"):
            self.criterion.register_buffer("global_step_views", torch.zeros((), dtype=torch.long, device=self.device))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator
    
    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        for epoch in range(self.start_epoch, self.n_epochs):
            if epoch % self.eval_interval == 0:
                self.logger.info(f"Evaluating epoch {epoch}...")
                val_loss = self.evaluate(epoch)
                self.save_best_checkpoint(val_loss, epoch)
                self.logger.info(f"Val loss: {val_loss}")
                self.logger.info(f"Evaluation complete.")
                torch.cuda.empty_cache()

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            self.t = time.time()
            self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch: self.save_model(epoch)
            torch.cuda.empty_cache()

        val_loss = self.evaluate(self.n_epochs)
        self.save_best_checkpoint(val_loss, self.n_epochs)

        self.save_model(self.n_epochs, is_final=True)
        torch.cuda.empty_cache()

    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            
            if str(self.criterion) == "LeJEPA":
                data["metadata"] = {k: v.to(self.device) for k,v in data["metadata"].items()}
                anysat_flag = False
                views = []
                for _ in range(self.n_global):
                    views.append(self.temporal_transform(data["image"]["optical"].to(self.device)))

                B, C, T, H, W = views[0].shape
                
                self.training_stats["data_time"].update(time.time() - end_time)

                with torch.autocast(
                    "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                ):
                    global_views = []
                    local_views = []
                    with torch.no_grad():
                        global_step_sync = all_reduce(self.criterion.global_step_views.clone(), op="MAX")
                        seed = global_step_sync.item()

                        g = self._get_generator(self.device, seed)

                        local_indexes = torch.randint(0, T, (self.n_local//self.n_global,), generator=g, device=self.device).long()
                        self.criterion.global_step_views.add_(1)

                    for view in views:
                        out, feature_maps, _, _ = self.model(view, batch_positions=data["metadata"])
                        out = self.model(out, projection=True)
                        global_views.append(out)
                        for local_index in range(self.n_local//self.n_global):
                            local_views.append(self.model(feature_maps[-1][:,local_indexes[local_index],:,:,:], projection=True))
                    
                    loss, inv, sigreg = self.compute_loss(global_views, local_views)

                    if loss == 0.0:
                        loss = loss + 0.0 * sum(p.sum() for p in self.model.module.parameters())

            else:
                anysat_flag = (str(self.criterion) == "AnySatJEPA")
                if str(self.criterion) == "UnifiedCoMMLoss":
                    # 1. Llevar el diccionario de imágenes al dispositivo
                    image_dict = {k: v.to(self.device) for k, v in data["image"].items()}
                    
                    # 2. Llevar metadata al dispositivo (doy, etc.)
                    metadata = None
                    if "metadata" in data and data["metadata"] is not None:
                        metadata = dict_to_device(data.get("metadata", {}), self.device)

                    # 3. Generar las vistas fuertemente aumentadas (Z' y Z'')
                    # temporal_transform maneja la consistencia espacio-temporal y multimodal automáticamente
                    aug1_img_dict = self.temporal_transform(image_dict)
                    aug2_img_dict = self.temporal_transform(image_dict)
                    
                    # 4. Calcular la pérdida CoMM Unificada
                    # Pasamos las imágenes limpias (para prototipos unimodales puros Z_1, Z_2)
                    # Y las imágenes aumentadas (para los prototipos de fusión Z', Z'')
                    outputs = self.criterion(
                        self.model,
                        orig_img_dict=image_dict, 
                        aug1_img_dict=aug1_img_dict, 
                        aug2_img_dict=aug2_img_dict, 
                        batch_positions=metadata
                    )
                    
                    loss = outputs["loss"]
                else:
                    data["metadata"] = {k: v.to(self.device) for k,v in data["metadata"].items()}
                    img = self.temporal_transform(data["image"]["optical"].to(self.device))
                    with torch.autocast("cuda", enabled=self.enable_mixed_precision, dtype=self.precision):
                        loss = self.criterion(img, self.model, batch_positions=data["metadata"])
                
            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.training_stats['loss'].update(loss.item())
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_inv": inv,
                        "train_sigreg": sigreg,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    } if str(self.criterion) == "LeJEPA" else {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        }, 
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            if anysat_flag: self.criterion.module.update_teacher_ema(self.model, momentum=0.996)

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        return

    @torch.no_grad()
    def evaluate(self, epoch: int):
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.eval()

        if str(self.criterion) == "LeJEPA":

            total_epoch_loss = 0.0
            total_inv = 0.0
            total_sigreg = 0.0
            
            end_time = time.time()
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc=f"Evaluating Epoch {epoch}", disable=self.rank != 0)):

                data["metadata"] = {k: v.to(self.device) for k,v in data["metadata"].items()}
                image = data["image"]["optical"].to(self.device)
                B, C, T, H, W = image.shape

                self.training_stats["data_time"].update(time.time() - end_time)

                with torch.autocast(
                    "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                ):

                    local_indexes = torch.linspace(0, T-1, self.n_local//self.n_global).long()
                    out, feature_maps, _, _ = self.model(image, batch_positions=data["metadata"])
                    global_views = [self.model(out, projection=True)]
                    local_views = [self.model(feature_maps[-1][:,local_index,:,:,:], projection=True) for local_index in local_indexes]

                    batch_loss, inv, sigreg = self.compute_loss(global_views, local_views)

                total_epoch_loss += batch_loss.item()
                total_inv += inv
                total_sigreg += sigreg
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

            final_val_loss = total_epoch_loss / len(self.val_loader)
            final_inv = total_inv / len(self.val_loader)
            final_sigreg = total_sigreg / len(self.val_loader)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                metrics_tensor = torch.tensor([final_val_loss, final_inv, final_sigreg], device=self.device)
                torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)
                final_val_loss = metrics_tensor[0].item()
                final_inv = metrics_tensor[1].item()
                final_sigreg = metrics_tensor[2].item()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "val_loss": final_val_loss,
                        "val_inv": final_inv,
                        "val_sigreg": final_sigreg,
                        "epoch": epoch
                    },
                    step = epoch * len(self.train_loader)
                )
                
            return final_val_loss
        elif str(self.criterion) == "UnifiedCoMMLoss":

            total_epoch_loss = 0.0
            total_fusion = 0.0
            total_z1_align = 0.0
            total_z2_align = 0.0
            
            end_time = time.time()
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc=f"Evaluating Epoch {epoch}", disable=self.rank != 0)):

                image_dict = {k: v.to(self.device) for k, v in data["image"].items()}
                metadata = None
                if "metadata" in data and data["metadata"] is not None:
                    metadata = dict_to_device(data.get("metadata", {}), self.device)

                self.training_stats["data_time"].update(time.time() - end_time)

                with torch.autocast(
                    "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                ):
                    aug1_img_dict = self.temporal_transform(image_dict)
                    aug2_img_dict = self.temporal_transform(image_dict)
                    
                    outputs = self.criterion(
                        self.model,
                        orig_img_dict=image_dict, 
                        aug1_img_dict=aug1_img_dict, 
                        aug2_img_dict=aug2_img_dict, 
                        batch_positions=metadata
                    )
                    
                total_epoch_loss += outputs["loss"].item()
                total_fusion += outputs["loss_fusion"].item()
                total_z1_align += outputs["loss_z1_align"].item()
                total_z2_align += outputs["loss_z2_align"].item()
                
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                end_time = time.time()

            final_val_loss = total_epoch_loss / len(self.val_loader)
            final_fusion = total_fusion / len(self.val_loader)
            final_z1_align = total_z1_align / len(self.val_loader)
            final_z2_align = total_z2_align / len(self.val_loader)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                metrics_tensor = torch.tensor([final_val_loss, final_fusion, final_z1_align, final_z2_align], device=self.device)
                torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)
                final_val_loss = metrics_tensor[0].item()
                final_fusion = metrics_tensor[1].item()
                final_z1_align = metrics_tensor[2].item()
                final_z2_align = metrics_tensor[3].item()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "val_loss": final_val_loss,
                        "val_fusion_loss": final_fusion,
                        "val_z1_align": final_z1_align,
                        "val_z2_align": final_z2_align,
                        "epoch": epoch
                    },
                    step = epoch * len(self.train_loader)
                )
                
            return final_val_loss
        else:
            total_epoch_loss = 0
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc=f"Evaluating Epoch {epoch}", disable=self.rank != 0)):
                data["metadata"] = {k: v.to(self.device) for k,v in data["metadata"].items()}
                
                if str(self.criterion) == "CoMMLoss":
                    img_dict = {k: v.to(self.device) for k, v in data["image"].items()}
                    img_dict = self.temporal_transform(img_dict)
                    with torch.autocast("cuda", enabled=self.enable_mixed_precision, dtype=self.precision):
                        total_epoch_loss += self.criterion(img_dict, self.model, batch_positions=data["metadata"]).item()
                else:
                    img = self.temporal_transform(data["image"]["optical"].to(self.device))
                    with torch.autocast("cuda", enabled=self.enable_mixed_precision, dtype=self.precision):
                        total_epoch_loss += self.criterion(img, self.model, batch_positions=data["metadata"]).item()

                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

            final_val_loss = total_epoch_loss / len(self.val_loader)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                metrics_tensor = torch.tensor([final_val_loss], device=self.device)
                torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)
                final_val_loss = metrics_tensor.item()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "val_loss": final_val_loss,
                        "epoch": epoch
                    },
                )
                
            return final_val_loss
            
    @torch.no_grad()
    def temporal_transform(self, x: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(x, dict):
            keys = list(x.keys())
            tensors = [x[k] for k in keys]
            splits = [t.shape[1] for t in tensors]
            
            x_concat = torch.cat(tensors, dim=1)
            
            x_out = self._do_temporal_transform(x_concat)
            
            x_split = torch.split(x_out, splits, dim=1)
            return {k: v for k, v in zip(keys, x_split)}
        else:
            return self._do_temporal_transform(x)

    def _do_temporal_transform(self, x: torch.Tensor):
        B, C, Temp, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B*Temp, C, H, W)

        x_out = torch.empty((B,C,Temp,H,W), device=x.device)

        for b in range(B):
            x_b = x[b*Temp:(b+1)*Temp]

            sample = self.transform({"image": x_b})

            x_b = sample["image"].permute(1, 0, 2, 3)

            x_out[b] = x_b

        return x_out

    def get_checkpoint(self, epoch: int) -> dict[str, dict | int]:
        """Create a checkpoint dictionary, containing references to the pytorch tensors.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict[str, dict | int]: checkpoint dictionary.
        """
        checkpoint = {
            "model": self.model.module.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        } if str(self.criterion) != "AnySatJEPA" else {
            "model": self.model.module.state_dict(),
            "criterion": self.criterion.module.teacher.state_dict(),
            "predictor": self.criterion.module.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        return checkpoint

    def save_model(
        self,
        epoch: int,
        is_final: bool = False,
        is_best: bool = False,
        checkpoint: dict[str, dict | int] | None = None,
    ):
        """Save the model checkpoint.

        Args:
            epoch (int): number of the epoch.
            is_final (bool, optional): whether is the final checkpoint. Defaults to False.
            is_best (bool, optional): wheter is the best checkpoint. Defaults to False.
            checkpoint (dict[str, dict  |  int] | None, optional): already prepared checkpoint dict. Defaults to None.
        """
        if self.rank != 0:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        suffix = "_best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{suffix}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        return

    def load_model(self, resume_path: str | pathlib.Path) -> None:
        """Load model from the checkpoint.

        Args:
            resume_path (str | pathlib.Path): path to the checkpoint.
        """
        model_dict = torch.load(resume_path, map_location=self.device, weights_only=False)
        if "model" in model_dict:
            self.model.module.load_state_dict(model_dict["model"])
            self.criterion.load_state_dict(model_dict["criterion"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"]
        else:
            self.model.module.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(
            f"Loaded model from {resume_path}. Resume training from epoch {self.start_epoch}"
        )

    def save_best_checkpoint(
        self, loss: float, epoch: int
    ) -> None:
        """Update the best checkpoint according to the loss.

        Args:
            eval_metrics (dict[float, list[float]]): metrics computed on the validation set.
            epoch (int): number of the epoch.
        """
        if self.best_metric_comp(loss, self.best_metric):
            self.best_metric = loss
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    def compute_loss(self, global_views: torch.Tensor, local_views: torch.Tensor) -> torch.Tensor:
        """Compute the loss"""
        return self.criterion(global_views, local_views)

    def log(self, batch_idx: int, epoch) -> None:
        """Log the information.

        Args:
            batch_idx (int): number of the batch.
            epoch (_type_): number of the epoch.
        """
        left_batch_this_epoch = self.batch_per_epoch - batch_idx
        left_batch_all = (
            self.batch_per_epoch * (self.n_epochs - epoch - 1) + left_batch_this_epoch
        )
        left_time_this_epoch = sec_to_hm(
            left_batch_this_epoch * self.training_stats["batch_time"].avg
        )

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        self.logger.info(basic_info)