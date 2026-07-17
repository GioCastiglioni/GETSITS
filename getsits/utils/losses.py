import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp
import math
import copy
from multiprocessing import Value
from getsits.encoders.vit import Block
from torch.nn.init import trunc_normal_
from getsits.encoders.pos_embed import get_2d_sincos_pos_embed_with_scale

class FocalLossSoftMax(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -1,
    ):
        super(FocalLossSoftMax, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # input: [B, C, H, W]
        # targets: [B, H, W]
        
        log_p = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(log_p, targets, reduction='none', ignore_index=self.ignore_index)
        
        with torch.no_grad():
            p = torch.exp(log_p)
            # targets: [B, H, W] -> [B, 1, H, W]
            gathered_p = torch.gather(p, dim=1, index=targets.unsqueeze(1).clamp(min=0))
            gathered_p = gathered_p.squeeze(1)
        
        focal_weight = (1.0 - gathered_p) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            return loss[mask].mean()
        
        return loss.mean()
    
    def __str__(self):
        return 'FocalLoss'

class LeJEPA(nn.Module):
    def __init__(
            self,
            num_slices,
            lamb=0.05,
            knots=17,
        ):
        super(LeJEPA, self).__init__()
        self.num_slices = num_slices
        self.lamb = lamb
        self.knots = knots
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=self.knots), num_slices=self.num_slices)

    def forward(self, global_views, local_views):
        B, K = global_views[0].shape
        all_views_list = global_views + local_views
        all_views_tensor = torch.stack(all_views_list)
        global_centroid = torch.stack(global_views).mean(dim=0)
        inv = (global_centroid - all_views_tensor).square().mean()
        sigreg = self.sigreg(all_views_tensor)
        return (1-self.lamb)*inv + self.lamb*sigreg, inv.item(), sigreg.item()
    
    def __str__(self):
        return 'LeJEPA'


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        op_enum = dist.ReduceOp.AVG if op.upper() == "AVG" else dist.ReduceOp.SUM
        return functional_all_reduce(x, op=op_enum)
    else:
        return x


class SlicingUnivariateTest(torch.nn.Module):

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats
        
class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if dist.is_available() and dist.is_initialized():
            torch.distributed.nn.functional.all_reduce(
                x, torch.distributed.ReduceOp.AVG
            )
        return x

    @property
    def world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point for linear spacing methods.
            Only used for 'trapezoid' and 'simpson' integration. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd for
            'simpson' integration. For 'gauss-hermite', this determines the number
            of positive nodes. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule with linear spacing over [0, t_max]
            Default: 'trapezoid'.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - For 'gauss-hermite', nodes and weights are adapted from the standard
          Gauss-Hermite quadrature to integrate against exp(-t²).
        - Supports distributed training via all_reduce operations.
    """

    def __init__(
        self, t_max: float = 5, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N * self.world_size
    

#-------------------------------------------------------------------------------------------------------------------------

def apply_masks(x, masks):
    all_x = []
    for m in masks:
        # Expand mask indices to match the feature dimension (D)
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        
        # Gather selected tokens along the sequence dimension (dim=1)
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
        
    return torch.cat(all_x, dim=0)

def repeat_interleave_batch(x, batch_size, repeat):
    N = len(x) // repeat
    x = torch.cat([
        torch.cat([x[i*batch_size : (i+1)*batch_size] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

class MaskCollator(object):
    def __init__(
        self,
        input_size=(6, 6),
        patch_size=1,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=6,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  
        self.allow_overlap = allow_overlap  
        self._itr_counter = Value('i', -1)  

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if tries > 20:
                        raise ValueError('Could not find a valid mask')
        mask = mask.squeeze()
        if mask.dim() == 0:
            mask = mask.unsqueeze(0)
        
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        
        return mask, mask_complement

    def __call__(self, x):
        B = len(x)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            if self.allow_overlap:
                acceptable_regions = None

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
            
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        device = x.device
        for i, tensor in enumerate(collated_masks_pred):
            collated_masks_pred[i] = tensor.to(device)

        for i, tensor in enumerate(collated_masks_enc):
            collated_masks_enc[i] = tensor.to(device)
            
        return collated_masks_enc, collated_masks_pred

class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        embed_dim=192,
        predictor_embed_dim=96,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        scale: int = 1,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        
        self.predictor_pos_embed = get_2d_sincos_pos_embed_with_scale(
                                        predictor_embed_dim,
                                        int(num_patches**0.5),
                                        scale,
                                        cls_token=False,
                                    )
        
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
            
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1).to(x.device)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1).to(x.device)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x, pred_tokens], dim=1)
        
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        x = x[:, (N_ctxt + 1):]
        x = self.predictor_proj(x)

        return x

class AnySatJEPALoss(nn.Module):
    def __init__(
        self,
        student_encoder,
        latent_shape=(14, 14),
        embed_dim=192,
        predictor_embed_dim=96,
        predictor_depth=6,
        predictor_num_heads=4,
        scale=1,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=0,
        allow_overlap=False
    ):
        super().__init__()
        
        self.teacher = copy.deepcopy(student_encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False

        num_patches = latent_shape[0] * latent_shape[1]
        
        self.predictor = VisionTransformerPredictor(
            num_patches=num_patches,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            scale=scale
        )
        
        self.mask_collator = MaskCollator(
            input_size=latent_shape,
            patch_size=1,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=nenc,
            npred=npred,
            min_keep=min_keep,
            allow_overlap=allow_overlap
        )

    def forward(self, inputs, student_encoder, batch_positions=None):
        mask_enc, mask_pred = self.mask_collator(inputs)

        with torch.no_grad():
            h_teacher, _, _, _ = self.teacher(inputs, batch_positions=batch_positions)
            
            if h_teacher.ndim == 4:
                h_teacher = h_teacher.flatten(2).transpose(1, 2)
            
            h_teacher = F.layer_norm(h_teacher, (h_teacher.size(-1),))
            h_targets = apply_masks(h_teacher, mask_pred)
            h_targets = repeat_interleave_batch(h_targets, len(h_teacher), len(mask_enc))

        h_student, _, _, _ = student_encoder(inputs, batch_positions=batch_positions) 
        if h_student.ndim == 4:
            h_student = h_student.flatten(2).transpose(1, 2)
            
        h_context = apply_masks(h_student, mask_enc)

        student_preds = self.predictor(h_context, mask_enc, mask_pred)

        loss = F.smooth_l1_loss(student_preds, h_targets)
        return loss

    @torch.no_grad()
    def update_teacher_ema(self, student_encoder, momentum=0.996):
        for param_student, param_teacher in zip(student_encoder.parameters(), self.teacher.parameters()):
            param_teacher.data.mul_(momentum).add_((1.0 - momentum) * param_student.detach().data)

    def __str__(self):
        return "AnySatJEPA"



class CoMMLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def info_nce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def augment(self, img_dict):
        import torchvision.transforms.v2 as transforms
        first_k = list(img_dict.keys())[0]
        spatial_size = img_dict[first_k].shape[-2:]
        
        aug = transforms.Compose([
            transforms.RandomResizedCrop(size=spatial_size, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        keys = list(img_dict.keys())
        tensors_2d = []
        splits = []
        
        for k in keys:
            v = img_dict[k]
            if v.dim() == 5:
                B, C, T, H, W = v.shape
                v_flat = v.reshape(B, C * T, H, W)
                tensors_2d.append(v_flat)
                splits.append(C * T)
            else:
                tensors_2d.append(v)
                splits.append(v.shape[1])
                
        concat_tensor = torch.cat(tensors_2d, dim=1)
        aug_tensor = aug(concat_tensor)
        split_tensors = torch.split(aug_tensor, splits, dim=1)
        
        augmented_dict = {}
        for i, k in enumerate(keys):
            v_split = split_tensors[i]
            v_orig = img_dict[k]
            if v_orig.dim() == 5:
                B, C, T, H, W = v_orig.shape
                augmented_dict[k] = v_split.reshape(B, C, T, H, W)
            else:
                augmented_dict[k] = v_split
                
        return augmented_dict
        
    def forward(self, img_dict, model, batch_positions=None):
        x_prime = self.augment(img_dict)
        x_double_prime = self.augment(img_dict)
        
        out_prime = model(x_prime, batch_positions=batch_positions)
        out_double_prime = model(x_double_prime, batch_positions=batch_positions)
        
        z_prime = model(out_prime[-1], projection=True)
        z_double_prime = model(out_double_prime[-1], projection=True)
        
        loss_comm_main = self.info_nce_loss(z_prime, z_double_prime)
        
        loss_i_sum = 0
        for mod in img_dict.keys():
            x_i = {}
            for k, v in img_dict.items():
                if k == mod:
                    x_i[k] = v
                else:
                    x_i[k] = torch.zeros_like(v)
            
            out_i = model(x_i, batch_positions=batch_positions)
            z_i = model(out_i[-1], projection=True)
            
            loss_i = 0.5 * (self.info_nce_loss(z_i, z_prime) + self.info_nce_loss(z_i, z_double_prime))
            loss_i_sum += loss_i
            
        total_loss = loss_comm_main + loss_i_sum
        return total_loss
        
    def __str__(self):
        return "CoMMLoss"
    

import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """
    Recopila tensores de todas las GPUs y soporta la retropropagación
    de gradientes a través de los procesos (Vital para InfoNCE DDP).
    """
    @staticmethod
    def forward(ctx, x):
        if not dist.is_available() or not dist.is_initialized():
            return x
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if not dist.is_available() or not dist.is_initialized():
            return grads[0]
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        # Retorna el gradiente que le corresponde a este proceso/GPU
        return all_gradients[dist.get_rank()]

def all_gather_with_grad(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = GatherLayer.apply(tensor)
    return torch.cat(gathered, dim=0)

class UnifiedCoMMLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8

    def _concat_dicts(self, dicts):
        out = {}
        for k in dicts[0].keys():
            if isinstance(dicts[0][k], torch.Tensor):
                out[k] = torch.cat([d[k] for d in dicts], dim=0)
            elif isinstance(dicts[0][k], dict):
                out[k] = self._concat_dicts([d[k] for d in dicts])
            else:
                out[k] = dicts[0][k]
        return out

    def infonce(self, z1, z2):
        N = len(z1)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        
        sim_zii = (z1 @ z1.T) / self.temperature 
        sim_zjj = (z2 @ z2.T) / self.temperature 
        sim_zij = (z1 @ z2.T) / self.temperature 
        
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
            
        log_sim_Z = F.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        return loss

    def forward(self, model, orig_img_dict, aug1_img_dict, aug2_img_dict, batch_positions=None):
        def isolate_modality(img_dict, target_mod=None):
            out = {}
            for k, v in img_dict.items():
                if target_mod is None or k == target_mod:
                    out[k] = v
                else:
                    out[k] = torch.zeros_like(v)
            return out

        z1_opt_dict = isolate_modality(aug1_img_dict, "optical")
        z1_sar_dict = isolate_modality(aug1_img_dict, "sar")
        z1_fus_dict = isolate_modality(aug1_img_dict, None)

        z2_opt_dict = isolate_modality(aug2_img_dict, "optical")
        z2_sar_dict = isolate_modality(aug2_img_dict, "sar")
        z2_fus_dict = isolate_modality(aug2_img_dict, None)

        merged_img_dict = self._concat_dicts([
            z1_opt_dict, z1_sar_dict, z1_fus_dict,
            z2_opt_dict, z2_sar_dict, z2_fus_dict
        ])
        
        merged_bp = None
        if batch_positions is not None:
            merged_bp = self._concat_dicts([batch_positions] * 6)

        z_all = model(merged_img_dict, batch_positions=merged_bp, return_projected=True)
        
        z_all = F.normalize(z_all, p=2, dim=-1)

        chunk_size_local = z_all.shape[0] // 6
        z1_opt_local, z1_sar_local, z1_proto_local, z2_opt_local, z2_sar_local, z2_proto_local = torch.split(z_all, chunk_size_local, dim=0)

        z1_opt = all_gather_with_grad(z1_opt_local)
        z1_sar = all_gather_with_grad(z1_sar_local)
        z1_proto = all_gather_with_grad(z1_proto_local)
        z2_opt = all_gather_with_grad(z2_opt_local)
        z2_sar = all_gather_with_grad(z2_sar_local)
        z2_proto = all_gather_with_grad(z2_proto_local)

        loss_opt = (self.infonce(z1_opt, z2_proto) + self.infonce(z2_opt, z1_proto)) / 2.0
        loss_sar = (self.infonce(z1_sar, z2_proto) + self.infonce(z2_sar, z1_proto)) / 2.0
        loss_fusion = self.infonce(z1_proto, z2_proto) 

        total_loss = (loss_opt + loss_sar + loss_fusion) / 3.0
        
        return {
            "loss": total_loss, 
            "loss_fusion": loss_fusion, 
            "loss_z1_align": loss_opt,
            "loss_z2_align": loss_sar
        }

    def __str__(self):
        return "UnifiedCoMMLoss"


class LeJEPA_CoMMLoss(nn.Module):
    def __init__(self, use_rbf=True, sigma_max=2.0, sigma_min=0.5, sigreg_weight=0.05):
        super().__init__()
        self.use_rbf = use_rbf
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma = sigma_max
        self.sigreg_weight = sigreg_weight
        
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=4096)

    def step(self, current_epoch, total_epochs):
        if self.use_rbf:
            self.sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * current_epoch / total_epochs))

    def k_sim(self, x, y):
        mse = F.mse_loss(x, y, reduction='none').mean(dim=-1) 
        
        if self.use_rbf:
            correntropy = torch.exp(-mse / (2 * self.sigma ** 2))
            return 1.0 - correntropy.mean()
        
        return mse.mean()

    def _concat_dicts(self, dicts):
        out = {}
        for k in dicts[0].keys():
            if isinstance(dicts[0][k], torch.Tensor):
                out[k] = torch.cat([d[k] for d in dicts], dim=0)
            elif isinstance(dicts[0][k], dict):
                out[k] = self._concat_dicts([d[k] for d in dicts])
            else:
                out[k] = dicts[0][k]
        return out

    def forward(self, model, orig_img_dict, aug1_img_dict, aug2_img_dict, batch_positions=None):
        def isolate_modality(img_dict, target_mod=None):
            out = {}
            for k, v in img_dict.items():
                if target_mod is None or k == target_mod:
                    out[k] = v
                else:
                    out[k] = torch.zeros_like(v)
            return out

        z1_opt_dict = isolate_modality(aug1_img_dict, "optical")
        z1_sar_dict = isolate_modality(aug1_img_dict, "sar")
        z1_fus_dict = isolate_modality(aug1_img_dict, None)

        z2_opt_dict = isolate_modality(aug2_img_dict, "optical")
        z2_sar_dict = isolate_modality(aug2_img_dict, "sar")
        z2_fus_dict = isolate_modality(aug2_img_dict, None)

        merged_img_dict = self._concat_dicts([
            z1_opt_dict, z1_sar_dict, z1_fus_dict,
            z2_opt_dict, z2_sar_dict, z2_fus_dict
        ])
        
        merged_bp = None
        if batch_positions is not None:
            merged_bp = self._concat_dicts([batch_positions] * 6)

        # [6B, D, H, W]
        out_spatial_list = model(merged_img_dict, batch_positions=merged_bp, return_projected=False)
        z_all_spatial = out_spatial_list[-1] 
        
        # [6B, D]
        z_all_bar = z_all_spatial.mean(dim=(2, 3))

        chunk_size_local = z_all_bar.shape[0] // 6
        (z1_opt_local, z1_sar_local, z1_proto_local, 
         z2_opt_local, z2_sar_local, z2_proto_local) = torch.split(z_all_bar, chunk_size_local, dim=0)

        z1_opt = all_gather_with_grad(z1_opt_local)
        z1_sar = all_gather_with_grad(z1_sar_local)
        z1_proto = all_gather_with_grad(z1_proto_local)
        z2_opt = all_gather_with_grad(z2_opt_local)
        z2_sar = all_gather_with_grad(z2_sar_local)
        z2_proto = all_gather_with_grad(z2_proto_local)
        
        loss_opt = (self.k_sim(z1_opt, z2_proto) + self.k_sim(z2_opt, z1_proto)) / 2.0
        loss_sar = (self.k_sim(z1_sar, z2_proto) + self.k_sim(z2_sar, z1_proto)) / 2.0
        loss_cross = loss_opt + loss_sar
        
        loss_synergy = self.k_sim(z1_proto, z2_proto)
        
        z_all_bar_global = torch.cat([z1_opt, z1_sar, z1_proto, z2_opt, z2_sar, z2_proto], dim=0)
        loss_sigreg = self.sigreg(z_all_bar_global)
        
        total_loss = loss_cross + loss_synergy + (self.sigreg_weight * loss_sigreg)
        
        return {
            "loss": total_loss, 
            "loss_cross": loss_cross,
            "loss_synergy": loss_synergy,
            "loss_sigreg": loss_sigreg
        }

    def __str__(self):
        return "LeJEPA_CoMMLoss"