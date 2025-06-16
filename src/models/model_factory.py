
import timm
import torch
from src.models.ResNeXt import  get_resnext
from torch import nn
import torchvision.models as models
from pathlib import Path
from src.models.ml_decoder import add_ml_decoder_head


def model_factory(model_name:nn.Module, n_class:int, freeze_layers:bool =True, 
                 percentage_freeze: int= 0, pretrained:bool= True, ml_decoder:bool =False):

    if  model_name =="resnext101_64x4d" :
        #model = timm.create_model('gluon_resnext101_64x4d', pretrained=True, num_classes=n_class)
        model = get_resnext(model_name = model_name, n_class= n_class, 
                             pretrained =pretrained)
        
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            freeze_weights(model, percentage_freeze )
    elif model_name =='resnext101_32x4d':
        #model = timm.create_model('gluon_resnext50_32x4d', pretrained=True, num_classes=n_class)
        model = get_resnext(model_name = model_name, n_class= n_class, 
                             pretrained =pretrained)
        
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            freeze_weights(model, percentage_freeze )
    elif model_name =='inceptionv4':
        model = timm.create_model('inception_v4', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights(model, percentage_freeze )
    elif model_name =='bit':
        model = timm.create_model('resnetv2_101x1_bitm', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )     
    elif model_name =='regnety':
        model = timm.create_model('regnety_002', pretrained=pretrained, num_classes=n_class)
    elif model_name =='nasnetalarge':
        model = timm.create_model('nasnetalarge', pretrained=False, num_classes=1001)
        model.load_state_dict(torch.load(str(Path(__file__).parent.parent.parent / 'models' / 'nasnetalarge-a1897284.pth')))
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, n_class)
    elif model_name =='vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_large_patch16_224':
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_huge_patch14_224_in21k':
        model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_tiny_patch16_224':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_small_patch32_224':
        model = timm.create_model('vit_small_patch32_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_patch32_224':
        model = timm.create_model('vit_base_patch32_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_tiny_patch16_224_in21k':
        model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_small_patch16_224_in21k':
        model = timm.create_model('vit_small_patch16_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_patch16_224_in21k':
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_patch8_224_in21k':
        model = timm.create_model('vit_base_patch8_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_patch32_224_in21k':
        model = timm.create_model('vit_base_patch32_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_r50_s16_224_in21k':
        model = timm.create_model('vit_base_r50_s16_224_in21k', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name =='vit_base_r50_s16_224':
        model = timm.create_model('vit_base_r50_s16_224', pretrained=pretrained, num_classes=n_class)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    elif model_name == 'nest': #https://arxiv.org/abs/2105.12723
        model = timm.create_model('jx_nest_base', pretrained=pretrained, num_classes=n_class) #, drop_rate = dropout_p, drop_path_rate =stochastic_depth_rate)
        if freeze_layers:
            freeze_weights_timm(model, percentage_freeze )
    else:
        raise NotImplemented(f"Model Architecture {model_name} not implemented")
    if ml_decoder:
        model = add_ml_decoder_head(model, num_classes=n_class, num_of_groups=n_class)
    return model

def freeze_weights(model: nn.Module, percentage_freeze: int):
    """Freeze model weights.
    This is done in two stages
    Stage-1 Freezing all the layers
    Stage-2 Unfreeze after specific block/layers

    Args:
        model (nn.model): model
        percentage_freeze (int): percentage of blocks of layers to freeze

    """    
    model_name = model.__class__.__name__
    # for _, param in model.named_parameters():
    #     param.requires_grad = False
  
    num_layers = int([name for name, _ in model.features.named_children()][-1]) 
    unfreeze_after =  int(num_layers * percentage_freeze)
    for name, child in model.features.named_children():
        if int(name) <= unfreeze_after:
            for param in child.parameters():
                param.requires_grad = False
    print(f'Frozen {unfreeze_after}/{num_layers} layers/blocks in {model_name}.') 


def freeze_weights_timm(model: nn.Module, percentage_freeze: int):      
    """Freeze model weights.
    This is done in two stages
    Stage-1 Freezing all the layers
    Stage-2 Unfreeze after specific block/layers

    Args:
        model (nn.model): model
        percentage_freeze (int): percentage of blocks of layers to freeze

    """  
    model_name = model.__class__.__name__    
    ct = 0
    num_layers =0
    for _, child in model.named_children():
        for _,params in child.named_parameters():
            num_layers+=1

    unfreeze_after =  int(num_layers * percentage_freeze)
    for _, child in model.named_children():     
        for _,params in child.named_parameters():
            if unfreeze_after > ct:
                params.requires_grad = False
            ct=ct+1   
    print(f'{model_name}- Frozen {unfreeze_after}/{num_layers} layers/blocks in {model_name}.')   


