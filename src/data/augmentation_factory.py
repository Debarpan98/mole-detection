from torchvision import transforms
from PIL import Image

def augmentationFactory(augmentation: str, size:int = 224, resized_crop_scale:float = 0.3, image_scale: int =256):
    """Data augmentation factory

    Args:
        augmentation (str): augmentation name
        size (int, optional): image desired shape. Defaults to 224.

    Returns:
        augmentation (transforms.Compose): data augmentations 
    """ 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augmentation =='random_crop':

        transform = transforms.Compose([
            transforms.Resize(image_scale,  Image.BICUBIC),
            transforms.RandomResizedCrop(size, scale=(resized_crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif augmentation == 'center_crop':

        transform =  transforms.Compose([
            transforms.Resize(image_scale, Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif augmentation == 'color_jitter':

        transform =  transforms.Compose([
            transforms.Resize(size, Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    elif augmentation == 'random_crop_jitter':

        transform =  transforms.Compose([
            transforms.Resize(image_scale, Image.BICUBIC),
            transforms.RandomResizedCrop(size, scale=(resized_crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])


    elif augmentation == 'randaugment': 
        transform =  transforms.Compose([
            transforms.Resize(image_scale, Image.BICUBIC),
            transforms.RandAugment(),
            transforms.RandomResizedCrop(size, scale=(resized_crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif augmentation == 'autoaugment': 
        transform =  transforms.Compose([
            transforms.Resize(image_scale, Image.BICUBIC),
            transforms.AutoAugment(policy =  transforms.autoaugment.AutoAugmentPolicy.SVHN),
            transforms.RandomResizedCrop(size, scale=(resized_crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    

    elif augmentation == 'noaugment':
        transform = transforms.Compose([
            transforms.Resize(image_scale, Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    else: 
        NotImplemented(f"Augmentaion {augmentation} not implemented")
    
    return transform