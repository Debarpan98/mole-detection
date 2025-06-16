import torch
import math
from src.models.cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def schedulerFactory(optimizer: torch.optim, name: str, lr:float, scheduler_steps :int, 
                     gamma:float, epochs: int, steps_per_epoch: int, warmup_epochs: int, 
                     cosine_cycle_epochs: int, cosine_cycle_decay: float,
                     onecycle_three_phase:bool = False) -> torch.optim.lr_scheduler:
    """Scheduler factory

    Args:
        optimizer (torch.optim): optimizer
        name (str): scheduler name
        lr (float): learnign rate
        step_size (int): step lr step size
        gamma (float): step lr gamma value
        epochs (int): number of epochs
        steps_per_epoch (int): number of images per loader

    Raises:
        NotImplemented: [description]

    Returns:
        torch.optim.lr_scheduler: [description]
    """    

    if name =='onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                        steps_per_epoch=steps_per_epoch,
                                                        epochs=epochs, three_phase=onecycle_three_phase
        )

    elif name =='steplr':
        total_steps = epochs * steps_per_epoch
        step_size = math.ceil(total_steps / scheduler_steps)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name =='cosine_annealing':
        # from https://timm.fast.ai/SGDR#warmup_t
        #scheduler =  CosineLRScheduler(optimizer, t_initial=cosine_cycle_epochs, cycle_decay=cosine_cycle_decay, 
                                     # warmup_t =warmup_epochs,  lr_min=1e-5, warmup_lr_init=1e-5, cycle_limit = 3)
        
        #scheduler = CosineLRScheduler(optimizer, t_initial=epochs, k_decay=0.9, warmup_t =warmup_epochs,  lr_min=1e-5, warmup_lr_init=1e-5, 
                                      # cycle_limit = cosine_cycle_limit)
        
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=cosine_cycle_epochs * steps_per_epoch, max_lr = lr, min_lr= 1e-5, 
                                                    warmup_steps=warmup_epochs * steps_per_epoch, gamma = cosine_cycle_decay )
        #torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,  T_mult =2, 
                                                                        # eta_min=1e-5, last_epoch=- 1)
    elif name == 'noscheduler':
        scheduler = None

    else:
        raise NotImplemented(f"Scheduler {name} not implemented")

    return scheduler