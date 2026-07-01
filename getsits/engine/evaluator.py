import logging
import os
import time
from pathlib import Path
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from PIL import Image


class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            criterion: torch.nn.Module,
            distribution: list,
            exp_dir: str | Path,
            device: torch.device,
            use_wandb: bool = False,
            dataset_name: str = 'pastis'
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.criterion = criterion
        self.exp_dir = exp_dir
        self.device = device
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.dataset_name = dataset_name

        self.use_wandb = use_wandb

        priors = torch.tensor(distribution, dtype=torch.float32)
        self.log_priors = torch.log(priors).to(self.device)

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            criterion: torch.nn.Module,
            distribution: list,
            exp_dir: str | Path,
            device: torch.device,
            use_wandb: bool = False,
            dataset_name: str = ""
    ):
        super().__init__(val_loader, criterion, distribution, exp_dir, device, use_wandb, dataset_name)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )
        total_loss = 0
        all_local_cms = [] 

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            data["metadata"] = {k: v.to(self.device) for k,v in data["metadata"].items()}
            image, target = data["image"], data["target"]
                
            image = {"v1": image["optical"].to(self.device)}
            target = target.to(self.device)
            logits = model(image["v1"], batch_positions=data["metadata"])
                
            if str(self.criterion) == "BalancedContrastiveLearning":
                loss_tensor = self.criterion.LC(logits, target)
            else:
                loss_tensor = self.criterion(logits, target)

            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            total_loss += loss_tensor.item()

            if model.module.segmentation:                
                if logits.shape[1] == 1:
                    pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
                else:
                    pred = torch.argmax(logits, dim=1)

                for b in range(target.shape[0]):
                    p_b = pred[b]
                    t_b = target[b]
                    valid_mask_b = t_b != self.ignore_index
                    p_b_valid, t_b_valid = p_b[valid_mask_b], t_b[valid_mask_b]
                    
                    count_b = torch.bincount(
                        (p_b_valid * self.num_classes + t_b_valid), minlength=self.num_classes ** 2
                    ).view(self.num_classes, self.num_classes)
                    
                    confusion_matrix += count_b
                    
                    if self.split == 'test':
                        all_local_cms.append(count_b.cpu())
                
                self.is_multilabel = False
                if self.split == 'test':
                    self.save_visualizations(image["v1"], target, pred, batch_idx, max_samples=2)
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                targets = target.float()
                    
                tp = (preds * targets).sum(dim=0)
                fp = (preds * (1 - targets)).sum(dim=0)
                fn = ((1 - preds) * targets).sum(dim=0)
                    
                confusion_matrix[0] += tp
                confusion_matrix[1] += fp
                confusion_matrix[2] += fn

                self.is_multilabel = True
            
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            
        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        
        metrics = self.compute_metrics(confusion_matrix.cpu())
        metrics["loss"] = total_loss / (self._get_dist_info()[1] * (batch_idx+1))

        if self.split == 'test' and hasattr(self, 'is_multilabel') and not self.is_multilabel:
            local_cms_tensor = torch.stack(all_local_cms).to(self.device)
            rank, world_size = self._get_dist_info()
            if world_size > 1:
                gathered_cms = [torch.zeros_like(local_cms_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_cms, local_cms_tensor)
                all_cms_tensor = torch.cat(gathered_cms, dim=0)
            else:
                all_cms_tensor = local_cms_tensor
                
            metrics["per_sample_cms"] = all_cms_tensor.cpu().numpy()
        
        self.log_metrics(metrics)
        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def save_visualizations(self, image_tensor, target_tensor, pred_tensor, batch_idx, max_samples=2):
        vis_dir = Path(f"~/workspace/datasets/visualizations/{self.exp_dir}")
        vis_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, 'palette'):
            np.random.seed(42)
            self.palette = np.random.randint(0, 255, size=(self.num_classes + 1, 3), dtype=np.uint8)
            self.palette[-1] = [0, 0, 0]

        B = image_tensor.shape[0]
        samples_to_save = min(B, max_samples)

        # 1. Definir la configuración estricta del encoder y obtener las del dataset
        encoder_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        dataset_bands = self.val_loader.dataset.bands['optical']
        
        # 2. Simular la lógica de BandFilter de tu data_preprocessor.py
        present_bands = [b for b in encoder_bands if b in dataset_bands]

        for b in range(samples_to_save):
            img = image_tensor[b].cpu().float().numpy()
            
            # Quitar la dimensión temporal de GETSITS: (C, T, H, W) -> (C, H, W)
            if img.ndim == 4:
                img = img[:, 0, :, :]
            
            # 3. Determinar qué bandas activas tiene el tensor actual
            if img.shape[0] == len(encoder_bands):
                # Se aplicó BandPadding, todas las bandas del encoder están presentes
                active_bands = encoder_bands
            else:
                # Solo se aplicó BandFilter, el tensor tiene la longitud de la intersección
                active_bands = present_bands

            # 4. Extraer los índices exactos de RGB dinámicamente
            try:
                r_idx = active_bands.index('B4')
                g_idx = active_bands.index('B3')
                b_idx = active_bands.index('B2')
                img = img[[r_idx, g_idx, b_idx], :, :]
            except ValueError:
                # Fallback de seguridad si el dataset no es óptico (ej. SAR)
                img = img[:3, :, :]
            
            # Convertir a formato de imagen (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            
            # 5. Normalización robusta para imágenes satelitales (Z-scores a RGB)
            img_min = np.percentile(img, 2, axis=(0, 1), keepdims=True)
            img_max = np.percentile(img, 98, axis=(0, 1), keepdims=True)
            
            img = (img - img_min) / (img_max - img_min + 1e-8)
            img = np.clip(img, 0, 1)
            
            # Corrección Gamma (0.8) para revivir los colores y eliminar neblina atmosférica
            img = img ** 0.8
            
            img_rgb = (img * 255).astype(np.uint8)

            tgt = target_tensor[b].cpu().numpy()
            prd = pred_tensor[b].cpu().numpy()

            # Mapear píxeles ignorados al final de la paleta
            valid_tgt = np.where((tgt == self.ignore_index) | (tgt < 0) | (tgt >= self.num_classes), self.num_classes, tgt)
            valid_prd = np.where((prd == self.ignore_index) | (prd < 0) | (prd >= self.num_classes), self.num_classes, prd)

            tgt_color = self.palette[valid_tgt]
            prd_color = self.palette[valid_prd]

            # --- NUEVA LÓGICA DE TRANSPARENCIA DINÁMICA ---
            # Nivel de opacidad para las clases válidas (ej. 50% transparente)
            alpha_base = 0.5 
            
            # Creamos una matriz de alpha donde:
            # - Si el índice es 0 (Background) o num_classes (Ignore), alpha = 1.0 (Solo se ve la imagen original)
            # - Si es cualquier otra clase objetivo, alpha = 0.5 (Se mezcla la imagen con el color)
            tgt_alpha = np.where((valid_tgt == 0) | (valid_tgt == self.num_classes), 1.0, alpha_base)[..., np.newaxis]
            prd_alpha = np.where((valid_prd == 0) | (valid_prd == self.num_classes), 1.0, alpha_base)[..., np.newaxis]

            # Aplicamos la combinación lineal (Alpha Blending Píxel a Píxel)
            prd_overlay = (img_rgb * prd_alpha + prd_color * (1 - prd_alpha)).astype(np.uint8)
            tgt_overlay = (img_rgb * tgt_alpha + tgt_color * (1 - tgt_alpha)).astype(np.uint8)
            # ----------------------------------------------

            combined_img = np.concatenate((prd_overlay, tgt_overlay), axis=1)

            filename = vis_dir / f"batch_{batch_idx:03d}_sample_{b}.png"
            Image.fromarray(combined_img).save(filename)

    def compute_metrics(self, confusion_matrix):
        if hasattr(self, 'is_multilabel') and self.is_multilabel:
            tp = confusion_matrix[0]
            fp = confusion_matrix[1]
            fn = confusion_matrix[2]

            # Avoid division by zero
            epsilon = 1e-6

            # Calculate metrics per class
            precision = tp / (tp + fp + epsilon) * 100
            recall = tp / (tp + fn + epsilon) * 100
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            iou = tp / (tp + fp + fn + epsilon) * 100

            miou = iou.mean().item()
            mf1 = f1.mean().item()
            macc = precision.mean().item() 

            iou = iou.cpu()
            f1 = f1.cpu()
            precision = precision.cpu()
            recall = recall.cpu()

            metrics = {
                "IoU": [iou[i].item() for i in range(len(iou))],
                "mIoU": miou,
                "F1": [f1[i].item() for i in range(len(f1))],
                "mF1": mf1,
                "mAcc": macc,
                "Precision": [precision[i].item() for i in range(len(precision))],
                "Recall": [recall[i].item() for i in range(len(recall))],
            }
            return metrics

        else:
            if self.ignore_index != -1:
                keep = torch.arange(confusion_matrix.size(0)) != self.ignore_index
                confusion_matrix = confusion_matrix[keep][:, keep]
            
            # Calculate IoU for each class
            intersection = torch.diag(confusion_matrix)
            union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
            iou = (intersection / (union + 1e-6)) * 100

            # Calculate precision and recall for each class
            precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
            recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

            # Calculate F1-score for each class
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            # Calculate mean IoU, mean F1-score, and mean Accuracy
            miou = iou.mean().item()
            mf1 = f1.mean().item()
            macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

            # Convert metrics to CPU and to Python scalars
            iou = iou.cpu()
            f1 = f1.cpu()
            precision = precision.cpu()
            recall = recall.cpu()

            # Prepare the metrics dictionary
            metrics = {
                "IoU": [iou[i].item() for i in range(confusion_matrix.size(0))],
                "mIoU": miou,
                "F1": [f1[i].item() for i in range(confusion_matrix.size(0))],
                "mF1": mf1,
                "mAcc": macc,
                "Precision": [precision[i].item() for i in range(confusion_matrix.size(0))],
                "Recall": [recall[i].item() for i in range(confusion_matrix.size(0))],
            }

            return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value, classes):
            header = f"------- {name} --------\n"
            metric_str = (
                "\n".join(
                    c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                    for c, num in zip(classes, values)
                )
                + "\n"
            )
            mean_str = (
                "-------------------\n"
                + "Mean".ljust(self.max_name_len, " ")
                + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        # Filter out ignored class if necessary
        if self.ignore_index != -1:
            filtered_classes = [c for i, c in enumerate(self.classes) if i != self.ignore_index]
            iou = [v for i, v in enumerate(metrics["IoU"]) if i != self.ignore_index]
            f1 = [v for i, v in enumerate(metrics["F1"]) if i != self.ignore_index]
            precision = [v for i, v in enumerate(metrics["Precision"]) if i != self.ignore_index]
            recall = [v for i, v in enumerate(metrics["Recall"]) if i != self.ignore_index]
        else:
            filtered_classes = self.classes
            iou = metrics["IoU"]
            f1 = metrics["F1"]
            precision = metrics["Precision"]
            recall = metrics["Recall"]

        iou_str = format_metric("IoU", iou, metrics["mIoU"], filtered_classes)
        f1_str = format_metric("F1-score", f1, metrics["mF1"], filtered_classes)

        precision_mean = torch.tensor(precision).mean().item()
        recall_mean = torch.tensor(recall).mean().item()

        precision_str = format_metric("Precision", precision, precision_mean, filtered_classes)
        recall_str = format_metric("Recall", recall, recall_mean, filtered_classes)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        loss_str = f"Validation Loss: {metrics['loss']:.4f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)
        self.logger.info(loss_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    f"{self.split}_loss": metrics["loss"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(filtered_classes, iou)
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(filtered_classes, f1)
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(filtered_classes, precision)
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(filtered_classes, recall)
                    },
                }
            )

    def _get_dist_info(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size