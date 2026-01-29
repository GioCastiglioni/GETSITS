import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float], warmup_epochs: int = 0, n_epochs: int = 100
) -> LRScheduler:
    
    """
    This module provides a utility function for creating a multi-step learning rate scheduler.

    Functions:
        MultiStepLR(optimizer: Optimizer, total_iters: int, lr_milestones: list[float]) -> LRScheduler
            Creates a MultiStepLR scheduler with specified milestones and decay factor.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_iters (int): The total number of iterations for training.
        lr_milestones (list[float]): A list of fractions representing the milestones at which the learning rate will be decayed.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler that decays the learning rate at specified milestones.
    """

    warmup_iters = warmup_epochs * (total_iters // n_epochs)

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_iters
    )

    scheduler_multi_step = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [(int((total_iters - warmup_iters) * r) + 1) for r in lr_milestones], gamma=0.1
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_multi_step],
        milestones=[warmup_iters]
    )

    return lr_scheduler


def CosineAnnealingLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float], warmup_epochs = 1, n_epochs=100
) -> LRScheduler:
    warmup_iters = warmup_epochs * (total_iters // n_epochs)
    s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int((total_iters-warmup_iters)*lr_milestones[0]), eta_min=1e-3*optimizer.param_groups[0]['lr'])
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[s1, s2],
        milestones=[warmup_iters]
    )

def CosineAnnealingWarmRestarts(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float], warmup_epochs = 1, n_epochs=100
) -> LRScheduler:

    warmup_iters = warmup_epochs * (total_iters // n_epochs)

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_iters
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        int((total_iters - warmup_iters)*lr_milestones[0]) + 1,
        eta_min = 1e-6
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_iters]
    )
    return lr_scheduler


