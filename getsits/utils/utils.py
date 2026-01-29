import os as os
import random
from pathlib import Path
from torch.optim.optimizer import Optimizer

import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as Fv
import torch.nn.functional as TF
from torchvision.transforms.v2 import InterpolationMode
from torchvision import tv_tensors


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to make flops calculator work
def prepare_input(input_res):
    image = {}
    x1 = torch.FloatTensor(*input_res)
    # input_res[-2] = 2
    input_res = list(input_res)
    input_res[-3] = 2
    x2 = torch.FloatTensor(*tuple(input_res))
    image["optical"] = x1
    image["sar"] = x2
    return dict(img=image)


def get_best_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_best.pth"))
    )

def get_final_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_final.pth"))
    )

class LARS(torch.optim.Optimizer):
    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        self.optimizer = optimizer
        self.eps = eps
        self.trust_coef = trust_coef

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @property
    def defaults(self):
        return self.optimizer.defaults

    @property
    def _state_dict_hooks(self):
        return self.optimizer._state_dict_hooks

    @property
    def _load_state_dict_pre_hooks(self):
        return self.optimizer._load_state_dict_pre_hooks

    @property
    def _load_state_dict_post_hooks(self):
        return self.optimizer._load_state_dict_post_hooks

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = p.data.norm()
                grad_norm = p.grad.data.norm()
                if param_norm > 0 and grad_norm > 0:
                    local_lr = self.trust_coef * param_norm / (grad_norm + self.eps)
                    p.grad.data.mul_(local_lr)
        self.optimizer.step(closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def register_state_dict_pre_hook(self, hook):
        return self.optimizer.register_state_dict_pre_hook(hook)

    def register_state_dict_post_hook(self, hook):
        return self.optimizer.register_state_dict_post_hook(hook)


class RandomChannelDropout(torch.nn.Module):
    def __init__(self, p=0.5, max_drop=1):
        super().__init__()
        self.p = p
        self.max_drop = max_drop

    def forward(self, img):
        """
        x: [T, C, H, W]
        """
        if "image" in img:
            x = img["image"].clone()
            if torch.rand(1).item() < self.p:
                C = x.shape[1]
                num_drop = torch.randint(1, self.max_drop + 1, ())
                drop_indices = torch.randperm(C)[:num_drop]
                x[:, drop_indices, :, :] = 0
                img["image"] = x

        return img


class ConsistentTransform(torch.nn.Module):
    def __init__(self, h_w=128, degrees=30):
        super().__init__()
        self.transforms = v2.Compose([
            v2.RandomResizedCrop(size=(h_w, h_w), scale=(0.2, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            ])

    def forward(self, sample):
        if "mask" in sample:
            sample = {"image": tv_tensors.Image(sample["image"]), "mask": tv_tensors.Mask(sample["mask"])}
            sample = self.transforms(sample)
            img, mask = torch.as_tensor(sample["image"]), torch.as_tensor(sample["mask"])

            img = self.add_gaussian_noise(img)

            return {"image": img, "mask": mask}
        else:
            sample = {"image": tv_tensors.Image(sample["image"])}
            sample = self.transforms(sample)
            img = torch.as_tensor(sample["image"])

            img = self.add_gaussian_noise(img)

            return {"image": img}
    
    def add_gaussian_noise(self, img, std_ratio=0.02):
        """
        Add Gaussian noise to a tensor image (C, H, W) or (B, C, H, W),
        with noise scaled by the image's dynamic range per channel.
        
        Args:
            img: Tensor image
            std_ratio: Noise std as a ratio of (max - min) per channel
        """
        # Compute min and max per channel (keep dims for broadcasting)
        dims = (-2, -1)  # spatial dims
        img_min = img.amin(dim=dims, keepdim=True)
        img_max = img.amax(dim=dims, keepdim=True)
        dynamic_range = img_max - img_min + 1e-8  # avoid zero division

        std = std_ratio * dynamic_range
        noise = torch.randn_like(img) * std
        noisy_img = img + noise
        
        # Clamp per channel using broadcasting (no .item())
        noisy_img = torch.max(noisy_img, img_min)
        noisy_img = torch.min(noisy_img, img_max)

        return noisy_img

class LeJEPATransform(torch.nn.Module):
    def __init__(self, h_w=128, degrees=30):
        super().__init__()
        self.degrees = degrees
        self.transforms = v2.Compose([
            v2.RandomResizedCrop(size=(h_w, h_w), scale=(0.3, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(11,11))], p=0.5)
            ])

    def forward(self, sample):
        if "mask" in sample:
            sample = {"image": tv_tensors.Image(sample["image"]), "mask": tv_tensors.Mask(sample["mask"])}
            sample = self.transforms(sample)
            img, mask = torch.as_tensor(sample["image"]), torch.as_tensor(sample["mask"])

            # Apply same rotation to both, with different interpolation modes
            angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
            img = self.rotate_with_reflection_padding(img, angle, is_mask=False)
            mask = self.rotate_with_reflection_padding(mask, angle, is_mask=True)

            img = self.add_gaussian_noise(img)

            return {"image": img, "mask": mask}
        else:
            # Apply same rotation to both, with different interpolation modes
            angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
            img=sample["image"]
            img = self.rotate_with_reflection_padding(img, angle, is_mask=False)

            sample = {"image": tv_tensors.Image(img)}
            sample = self.transforms(sample)
            img = torch.as_tensor(sample["image"])


            img = self.add_gaussian_noise(img)

            return {"image": img}

    def rotate_with_reflection_padding(self, img, angle, is_mask=False):
        """
        img: Tensor of shape (B, C, H, W) or (C, H, W)
        angle: float, rotation angle in degrees
        is_mask: if True, use nearest-neighbor interpolation
        """
        is_batched = img.dim() == 4
        if not is_batched:
            img = img.unsqueeze(0)

        B, C, H, W = img.shape
        diag = int((H**2 + W**2) ** 0.5)
        pad_h = (diag - H) // 2 + 1
        pad_w = (diag - W) // 2 + 1

        img_padded = TF.pad(img, [pad_w, pad_w, pad_h, pad_h], mode='reflect')

        # Choose interpolation
        interpolation = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR

        rotated = torch.stack([
            Fv.rotate(img_padded[i], angle, interpolation=interpolation)
            for i in range(B)
        ])
        cropped = torch.stack([Fv.center_crop(rotated[i], [H, W]) for i in range(B)])

        if not is_batched:
            return cropped.squeeze(0)
        return cropped
    
    def add_gaussian_noise(self, img, std_ratio=0.02):
        """
        Add Gaussian noise to a tensor image (C, H, W) or (B, C, H, W),
        with noise scaled by the image's dynamic range per channel.
        
        Args:
            img: Tensor image
            std_ratio: Noise std as a ratio of (max - min) per channel
        """
        # Compute min and max per channel (keep dims for broadcasting)
        dims = (-2, -1)  # spatial dims
        img_min = img.amin(dim=dims, keepdim=True)
        img_max = img.amax(dim=dims, keepdim=True)
        dynamic_range = img_max - img_min + 1e-8  # avoid zero division

        std = std_ratio * dynamic_range
        noise = torch.randn_like(img) * std
        noisy_img = img + noise
        
        # Clamp per channel using broadcasting (no .item())
        noisy_img = torch.max(noisy_img, img_min)
        noisy_img = torch.min(noisy_img, img_max)

        return noisy_img

