from cmath import inf
import os
import torch
import time
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import sklearn.utils.class_weight as class_weight
import time
from tqdm import tqdm

from pathlib import Path
from src.data.GCPDataset import GCPDataset
from typing import Tuple
from torch.autograd import Variable
from src.data.make_dataset import generate_split_data, generate_idx_dataset
from src.training.evaluate_model import evaluate_model
from src.models.optimizer_factory import optimizerFactory
from src.models.scheduler_factory import schedulerFactory
from src.models.model_factory import model_factory
from src.models.loss_factory import lossFactory
from src.utils.utils import setAllSeeds
from src.utils.gcs_utils import save_model, load_from_filename
from src.utils.mlflow_tracker import MlFlowTracker, log_avg_cv_metrics
from src.training.training_metrics import TrainingMetrics
from src.training.training_hyperparameters import TrainingHyperparameters, set_training_settings, generate_transforms
from src.training.compute_results import ResultsFactory
from src.data.dataset_creation_factory import DatasetCreationFactory
from src.data.augmentation_factory import augmentationFactory
from src.data.make_dataset import read_images_initial, get_dataset
from src.data.mixup import mixup_criterion, mixup_data

path_manager= None


def train_model(model: nn.Module,  dataloaders: dict, criterion: torch.nn.modules.loss, 
                optimizer: torch.optim, scheduler:torch.optim.lr_scheduler, accum_step_multiple:int=1, 
                device:torch.device = None, num_epochs: int=25, mixup:bool = False,
                mixup_alpha:float = 0.1, results:ResultsFactory=None, is_multi_label:bool = False, 
                str_name =None, early_stopping:bool=False, patience:int=5)-> Tuple[nn.Module, TrainingMetrics]:
    """Generic Training Function

    Args:
        model (nn.Module): untrained model
        dataloaders (dict): dictionary of dataloaders per mode
        criterion (torch.nn.modules.loss): loss function
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler
        device (torch.device, optional): device (cuda or cpu). Defaults to None.
        num_epochs (int, optional): numper of epochs. Defaults to 25.
        results (ResultsFactory, optional): object to compute the results (metrics). Defaults to None.
        is_multi_label (bool, optional): bool indicating if the the task is a multi-label classification. Defaults to False.

    Returns:
        nn.Module: trained model
        TrainingMetrics: object that contains all the training metrics
    """    
    since = time.time()
    metrics = TrainingMetrics(is_multi_label)
    train_epoch_loss =0.0
    stop = False
    if early_stopping:
        # Early stopping
        last_loss = float('inf')
        triggertimes = 0
        
        best_model_wts = model.state_dict()
    
    if 'val' in dataloaders.keys():
        modes=['train', 'val']
    else:
        modes=['train', 'test']
    
    for epoch in range(num_epochs):
        if stop:
            break
        for phase in modes:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
                
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            all_results = {'all_probs':[], 'all_labels': [], 'all_preds': []}

            for batch_idx, data in enumerate(tqdm(dataloaders[phase])):
                inputs, labels, _ = data
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))
                if mixup and phase=='train': # FROM-> https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                    outputs = model(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    if type(outputs) == tuple:
                         outputs, _ = outputs
                    loss = criterion(outputs, labels)

                results.update_results(all_results, outputs, labels)
                
                loss = loss / accum_step_multiple
                loss.backward()
                if phase == 'train' and (((batch_idx + 1) % accum_step_multiple == 0) 
                                         or (batch_idx + 1 == len(dataloaders[phase]))):
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                        metrics.learning_rates.extend(scheduler.get_last_lr())
                    optimizer.zero_grad(set_to_none=True)
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset.targets)
            metrics.get_losses_per_mode(phase).append(epoch_loss)
            printed_metric = metrics.add_scores(phase, all_results['all_labels'], all_results['all_preds'], all_results['all_probs'])
            
            if phase=='train':
                train_epoch_loss = epoch_loss
                train_printed_metric = printed_metric
            
            elif early_stopping and phase=='val':
                # early stopping
                if epoch_loss > last_loss:
                    triggertimes += 1
                    print('Trigger Times:', triggertimes)
                    if triggertimes >= patience:
                        print('Early stopping!\nStart to test process.')
                        model.load_state_dict(best_model_wts)
                        stop=True
                        break
                else:
                    best_model_wts = model.state_dict()
                   # print('trigger times: 0')
                    triggertimes = 0

                last_loss = epoch_loss
            
            if phase=='val' or phase=='test':
                print('Epoch: {}/{}: [{}]: {} Loss: {:.4f} {} |  {} Loss: {:.4f} {}'.format(
                epoch, num_epochs - 1, str_name, 'Train',  train_epoch_loss,  train_printed_metric,
                phase, epoch_loss,  printed_metric))
                
    time_elapsed = time.time() - since
    print('[{}]:Training complete in {:.0f}m {:.0f}s'.format(str_name,
        time_elapsed // 60, time_elapsed % 60))
    return model, metrics

def get_steps_per_epoch(num_batch, accum_step_multiple):
    steps_per_epoch =math.ceil(num_batch/accum_step_multiple)
    if steps_per_epoch==0:
        steps_per_epoch=1
    return steps_per_epoch

def train_internal(dataloaders: dict,   params: dict,  results:ResultsFactory=None) -> Tuple[nn.Module, TrainingMetrics]:
    """Trains model named 'model_name' and evaluate it
    Args:
        dataloaders (dict): dictionary of dataloaders
        params (dict): class with model hyperparameters
        results (ResultsFactory): results object (the results are computed differently between single-label task and multi-label task)

    Returns:
        model (nn.module): trained model
        metrics (TrainingMetrics) : experiment result metrics
    """    
    class_names = dataloaders['train'].dataset.targets
    model_conv = model_factory(params.model_name, len(class_names), freeze_layers =params.freeze_layers, 
                               percentage_freeze=params.percentage_freeze, pretrained= True, ml_decoder = params.ml_decoder)
    
    if len(params.gpus)>1:
        print("[Using GPU's]", params.gpus)
        model_conv = nn.DataParallel(model_conv, device_ids=params.gpus)
    
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    model_conv.to(device)
    
    class_weights=class_weight.compute_class_weight('balanced', classes= class_names, y = np.array(dataloaders['train'].dataset.targets))
    criterion = lossFactory(params.loss_function, class_weights=class_weights, focal_gamma = params.focal_gamma, 
                            focal_alpha = params.focal_alpha, asymmetric_gamma_neg = params.asymmetric_gamma_neg, 
                            asymmetric_gamma_pos = params.asymmetric_gamma_pos, asymmetric_clip= params.asymmetric_clip, device=device)
    optimizer_conv =optimizerFactory(model_conv, optimizer_name= params.optimizer_name, lr= params.learning_rate, 
                                     momentum=params.momentum, weight_decay=params.weight_decay)

    steps_per_epoch = get_steps_per_epoch(len(dataloaders['train']), params.accum_step_multiple)
    exp_lr_scheduler = schedulerFactory(optimizer=optimizer_conv, name=params.scheduler_name, lr=params.learning_rate, 
                                        scheduler_steps=params.scheduler_steps, gamma= params.gamma, epochs=params.epochs,
                                        steps_per_epoch= steps_per_epoch, warmup_epochs = params.warmup_epochs, 
                                        cosine_cycle_epochs = params.cosine_cycle_epochs,
                                         cosine_cycle_decay = params.cosine_cycle_decay,onecycle_three_phase= params.onecycle_three_phase)

    model, metrics = train_model(model_conv, dataloaders, criterion, optimizer_conv, exp_lr_scheduler,
                                 params.accum_step_multiple, device=device, num_epochs=params.epochs, mixup= params.mixup,
                                 mixup_alpha = params.mixup_alpha,  results=results,  is_multi_label = params.multi_label,
                                  str_name= params.str_name, early_stopping = params.early_stopping,
                                 patience= params.patience)
    return  model, metrics

def train_without_cross_validation(df_dataset: pd.DataFrame, params: dict, transform:dict):
    """Train model - not the cross validation training

    Args:
        df_dataset (pd.DataFrame): dataset dataframe
        params (dict): dictionary of hyperparameters and settings
        transform (dict): dictionary of data augmentations - for the test and train set
    """    
    mlflow_tracker = MlFlowTracker(params.model_name)  
    mlflow_tracker.log_params(params.params_dict)     
    mlflow_tracker.log_pd_file(df_dataset, f'{params.str_name}-dataset.csv')
    mlflow_tracker.log_json_file(params.__dict__ ,f'{params.str_name}_params.json') 

    dataset_creator = DatasetCreationFactory().create_dataset(is_multi_label=params.multi_label,  is_binary_classification=params.binary_classification,
                                                              check_ratio= params.check_ratio, subsamble_set= params.subsamble_set, 
                                                              skin_ratio_threshold =params.skin_ratio_threshold, subsample_max_values = params.subsample_max_values,
                                                              subsample_max_per_disease= params.subsample_max_per_disease, )

    dataloaders, image_datasets = generate_split_data(df_dataset = df_dataset, 
                                                mask_skin= params.mask_skin,
                                                transform=transform,
                                                classes=params.classes,
                                                random_state = params.random_state,
                                                batch_size = params.batch_size, 
                                                dataset_creator=dataset_creator, 
                                                num_workers = params.num_workers,
                                                test_set_from_file =params.test_set_from_file,
                                                test_set_path= params.test_set_path,
                                                val_set_from_file = params.val_set_from_file,
                                                val_set_path= params.val_set_path,
                                                train_set_from_file = params.val_set_from_file,
                                                train_set_path= params.train_set_path,
                                                val_set=params.val_set,
                                                one_tag=params.one_tag)

    results= ResultsFactory().create_task(params.multi_label, threshold=params.multi_label_threshold, label_encoder=image_datasets["train"].le)
    mlflow_tracker.log_pickle_file(image_datasets["train"].le, f'{params.str_name}-label_encoder.pkl')
    model, metrics  = train_internal(dataloaders = dataloaders, params= params, results=results) 
    if params.save_model:
        mlflow_tracker.save_model(model)
    #    save_model(model, params.gcs_bucket,  Path(params.path_bucket), mlflow_id = mlflow_tracker.run_id)
    metrics.save_metrics_mlflow(mlflow_tracker, params.str_name)    
    evaluate_model(model = model, model_name = params.str_name, mlflow_tracker = mlflow_tracker, dataloaders = dataloaders, label_encoder=image_datasets["train"].le, results=results)


def train_cross_validation(df_dataset: pd.DataFrame, params: dict, transform:dict,  k_folds:int =5):
    """Cross Validation Training.

    Args:
        df_dataset (pd.DataFrame): dataframe dataset
        params (dict): dictionary of hyperparameters
        transform (dict): dictionary of transformations (test and train)
        k_folds (int, optional): number of folds . Defaults to 5.
    """
    dataset_creator = DatasetCreationFactory().create_dataset(is_multi_label=params.multi_label,  is_binary_classification=params.binary_classification,
                                                              check_ratio= params.check_ratio, subsamble_set= params.subsamble_set, 
                                                              skin_ratio_threshold =params.skin_ratio_threshold, subsample_max_values = params.subsamples_max_values,
                                                              subsample_max_per_disease= params.subsample_max_per_disease
                                                              )
    kfold , np_labels = dataset_creator.create_folds(df_dataset, params.classes, k_folds, random_state=params.random_state)
    start_time = time.time()
    np_dataset = df_dataset.to_numpy()
    run_ids=[]
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np_dataset,np_labels)):
        print(f'FOLD {fold+1}-{params.str_name}')
        print('--------------------------------')
        mlflow_tracker = MlFlowTracker(params.model_name)  
        run_ids.append(mlflow_tracker.run_id)
        mlflow_tracker.log_json_file(params.__dict__ ,f'{params.str_name}_params.json') 
        
        mlflow_tracker.log_params(params.params_dict) 
        mlflow_tracker.log_param(fold, 'fold')    
        dataloaders, image_datasets = generate_idx_dataset(transform, df_dataset, params.mask_skin, params.classes,
                                                           train_ids, test_ids, batch_size=params.batch_size, 
                                                           multi_label=params.multi_label,num_workers =params.num_workers)

        results= ResultsFactory().create_task(params.multi_label, threshold=params.multi_label_threshold, label_encoder=image_datasets["train"].le)
        model, metrics  = train_internal(dataloaders = dataloaders, params= params, results=results) 
        metrics.save_metrics_mlflow(mlflow_tracker, params.str_name)    
        evaluate_model(model = model, mlflow_tracker = mlflow_tracker, dataloaders = dataloaders, 
                   label_encoder=image_datasets["train"].le, results=results)
        print('--------------------------------')
    
    print(f'Computing average over {k_folds} folds')
    total_time = time.time() - start_time
    avg_time_fold = total_time/k_folds
    overall_mlflow_tracker = MlFlowTracker(params.model_name) 
     
    overall_mlflow_tracker.log_metric(avg_time_fold, 'avg_time_fold')
    overall_mlflow_tracker.log_params(params.params_dict)     
    overall_mlflow_tracker.log_param(k_folds, 'k_folds')
    overall_mlflow_tracker.log_param(run_ids, 'run_ids')
    overall_mlflow_tracker.log_param(params.random_state, 'seed')
    overall_mlflow_tracker.log_pd_file(df_dataset, 'dataset.csv')
    log_avg_cv_metrics(mlflow_tracker.experiment, run_ids, overall_mlflow_tracker)
    

def train(model_name:str, config:dict)->None:
    """Train model named 'model_name'

    Args:
        model_name (str): name of the model to train
        config (dict): dictionary of hyperparameters and settings
    """
    global path_manager

    k_folds = config.get('k_folds')
    is_cross_validation = config.get('is_cross_validation')
    params = TrainingHyperparameters(model_name, config.get('model')[model_name])
    set_training_settings(config, params)
    
    setAllSeeds(params.random_state)
    transform =generate_transforms(params)
    df = get_dataset(config)
    df = read_images_initial(df, params.classes, params.binary_classification,params.multi_label, params.merge_acne)

    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_gpus

    if is_cross_validation:
        train_cross_validation(df_dataset=df, params=params, transform=transform, k_folds=k_folds)
    else:
        train_without_cross_validation(df_dataset= df, params=params,  transform=transform)

    
def predict(config:dict, df: pd.DataFrame)->None:
    """Functions that get all the predictions on all images (full dataset).
    The predictions are saved in Mlflow under experiment: 'ALL-Predictions'

    Args:
        config (dict): settings
        df (pd.DataFrame): dataset (full dataset)
    """

    model_name = config.get('predict_model')
    params = TrainingHyperparameters(model_name, config.get('model')[model_name])
    set_training_settings(config, params)
    
    if params.binary_classification:
        df['not_acne'] =  ~df['acne']

    dataset = GCPDataset(df_dataset=df,  classes=params.classes,transform= augmentationFactory('noaugment', params.size) , mutli_label=params.multi_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,shuffle=True, num_workers=params.num_workers  )
    mlflow_tracker = MlFlowTracker(params.model_name)  
    mlflow_tracker.log_param(param=config.get('predict_model_name'), name= 'model_name')
    mlflow_tracker.log_params(params.params_dict)     
    mlflow_tracker.log_pd_file(df, f'{params.str_name}-dataset.csv')
    mlflow_tracker.log_json_file(params.__dict__ ,f'{params.str_name}_params.json') 
    
    # load model
    model = model_factory(params.model_name, len(params.classes), freeze_layers =params.freeze_layers, 
                               percentage_freeze=params.percentage_freeze, pretrained= True)
    model.load_state_dict(torch.load(config.get('predict_model_name')))
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    model.to(device)
    results= ResultsFactory().create_task(params.multi_label, threshold=params.multi_label_threshold, label_encoder=dataset.le)
    evaluate_model(model = model, model_name = params.str_name, mlflow_tracker = mlflow_tracker, dataloaders =  dataloader, 
                   label_encoder=dataset.le, results=results)
