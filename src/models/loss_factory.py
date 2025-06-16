
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.focal_loss import FocalLoss
from src.models.asymmetric_loss import AsymmetricLossOptimized, ASLSingleLabel

def lossFactory(loss_name: str, class_weights: dict, focal_gamma: float =0., focal_alpha: float=1, 
                asymmetric_gamma_neg: int=4, asymmetric_gamma_pos:int = 1, asymmetric_clip: float= 0.05, device: torch.device = None) -> torch.nn: 
    """Factory Pattern to get loss function

    Args:
        loss_name (str): loss name
        class_weights (dict): class weights

    Raises:
        NotImplemented: loss function not implemented

    Returns:
        torch.nn: loss function
    """
    if loss_name =='ce':
        loss = nn.CrossEntropyLoss()
    elif loss_name =='weighted_ce':
        loss = get_weighted_cross_entropy(class_weights, device)
    elif loss_name=='bce':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name=='focal':
        loss= FocalLoss( gamma = focal_gamma, alpha= focal_alpha)
    elif loss_name=='asymmetric':
        loss= AsymmetricLossOptimized(gamma_neg= asymmetric_gamma_neg,  gamma_pos =  asymmetric_gamma_pos, clip= asymmetric_clip)
    elif loss_name=='asymmetric_single_label':
        loss=  ASLSingleLabel(gamma_neg= asymmetric_gamma_neg,  gamma_pos =  asymmetric_gamma_pos)
    else:
        raise NotImplemented(f"Loss function {loss_name} not implemented")

    return loss




def get_weighted_cross_entropy(class_weights, device=None):
    """
    Since the dataset is imbalanced, the loss function is weighted.
    This function creates the weighted loss function by computing the class weight.
    Args:
        dataset (torchvision.datasets.folder.ImageFolder): generated dataset
        trainloader (torch.utils.data.DataLoader): train dataloader
        indices (list): list of dataset training set indexes
    Returns:
        loss (nn.CrossEntropyLoss): loss function
    """  
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    return nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

