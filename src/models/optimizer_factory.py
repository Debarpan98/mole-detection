import torch

def optimizerFactory(model: torch.nn.Module, optimizer_name: str, lr: float, 
                     momentum: float, weight_decay: float) -> torch.optim:
    """Optimizer Factory

    Args:
        model (torch.nn.Module): model
        optimizer_name (str): optimizer name
        lr (float): learning rate
        momentum (float): momentum
        weight_decay (float): weight decay

    Raises:
        NotImplemented: optimizer not implemented

    Returns:
        torch.optim: optimizer
    """    

    if optimizer_name == 'adam':
        return torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, 
                                weight_decay=weight_decay, amsgrad=False
        )
    elif optimizer_name == 'sgd': 
       
        return torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr,
                                           momentum=momentum, weight_decay=weight_decay)

    else:
        raise NotImplemented(f"Optimizer {optimizer_name} not implemented")