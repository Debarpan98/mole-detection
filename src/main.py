import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import torch
import sys
import copy
import os
sys.path.append(str(Path(__file__).parent.parent))
import multiprocessing
from loguru import logger
from src.utils.gcs_utils import set_environement_variable

is_env_variables_set = set_environement_variable()
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/gauthies/sdd/sdd-general/dist/auth.json'


from src.utils import envs 
e=envs.Envs()
import src.utils.config_manager as config_manager
from src.utils.utils import logger_config
from src.training.train_model import train, predict
from pathlib import Path
from src.data.make_dataset import read_images_from_db
from src.utils.utils import  set_hyperparams

logger.configure(**logger_config())
multiprocessing.set_start_method('spawn', force=True)

path_manager=None

def start_processes(jobs):
    """Start all the processes 

    Args:
        jobs (list): list of processes
    """    
    for j in jobs:
        j.start()

    for j in jobs:
        try:
            print("JOINING TASK",j)
            j.join()
        except Exception as exc:
            print("EXCEPTION ON JOINING TASK")
            print(exc)
    print('TRAIN ASYNC DONE INSIDE')

    
def check_to_start_processes(jobs: list, num_devices: int, gpu_visible:int):
    """_summary_

    Args:
        jobs (list): list of processes
        num_devices (int): number of devices available
        gpu_visible (int): the last gpu that was used to start a process

    """    
    if (gpu_visible) ==num_devices:
        print(f'Starting {num_devices} process(es) on the {num_devices} GPUs available. ')
        start_processes(jobs)
        jobs = []
        gpu_visible=0
    return jobs, gpu_visible

def train_async(config: dict):
    """Starts the training of models asynchronously.
    """
    jobs = []
    modellist = config.get('modellist')
    num_devices = torch.cuda.device_count()
    num_cpus= os.cpu_count()
    num_workers= int(num_cpus /num_devices)
    print(f'We assign {num_workers} workers per dataloader.')
    gpu_visible = 0

    # for all models in modellist
    for model_name in modellist:
        params = config.get('model')[model_name]['params_settings']
        
        # for all parameters of models
        if params is not None:
            print(f'{model_name} model will be trained with {len(params)} sets of hyperparameters.')
            for index, key in enumerate(params):
                config_tmp = copy.deepcopy(config)
                params[key]['num_workers'] = num_workers
                num_gpus =  len(config_tmp.get('model')[model_name]['gpus'])  
                
                if num_gpus + gpu_visible > num_devices:
                    start_processes(jobs)
                    jobs = []
                    gpu_visible = 0
                
                for i in range(num_gpus):
                    if i ==0:
                        params[key]['visible_gpus'] = gpu_visible
                    else:
                        params[key]['visible_gpus'] = str(params[key]['visible_gpus'])+','+ str(gpu_visible)
                    gpu_visible+=1
                
                print(f"Putting {model_name} process {index} on GPU # {params[key]['visible_gpus']}.")
                params[key]['process'] = index
                set_hyperparams(model_name, config_tmp, params[key])
                model = multiprocessing.Process(target=train,  kwargs={"model_name":model_name, "config": config_tmp})
                jobs.append(model)
                jobs, gpu_visible = check_to_start_processes(jobs, num_devices, gpu_visible)
                
        # if model has only one set of parameters
        else:
            config_tmp  = copy.deepcopy(config)
            config_tmp.set_model_param(model_name, 'gpu_visible ', gpu_visible )
            print(f'Putting process {model_name} on GPU # {gpu_visible}.')
            model = multiprocessing.Process(target=train,  kwargs={"model_name":model_name, "config": config_tmp})
            jobs.append(model)
            gpu_visible+=1
            jobs, gpu_visible = check_to_start_processes(jobs, num_devices, gpu_visible)
    
    # the last processes to run
    if len(jobs)!=0:
        start_processes(jobs)
 

def train_all(config:dict):
    """This function starts the training.
        If there is more than one model to start, the training 
        is done in parallel. 
    """
   
    try:
        print("TRAIN ASYNC START")
        train_async(config)
        print("TRAIN ASYNC END")
    
    except Exception as exc:
        print("TRAIN ASYNC ERROR CATCH")
        print(exc)

def create_dataset(config:dict):
    """Create the dataset and save it in bucket
        This function should be called once in a while 
        to recreate the dataset (if there are new images in the dataset).
    """
    read_images_from_db(config)

def predict_all(config:dict):
    """This function is called to get the predictions on all the images.
    In the config file, there are two parameters you need to specify to make sure the predictions are
    made from the desired model. The parameters are:
        -predict_model: name of the model - for example 'bit'
        -predict_model_name: model state dicitonary path - for example 'model.pth'

    Args:
        config (dict): dictionary of settings
    """

    # df = read_images_initial(config)
    # load_from_filename(config.get('gcs_bucket'), path_manager.mlflow_gcs_db, path_manager.mlflow_local_db)
    # predict(config, df)
    # save_from_filename(config.get('gcs_bucket'), path_manager.mlflow_gcs_db, path_manager.mlflow_local_db)
    pass # TODO update function

if __name__ == '__main__':
    """
    Start the desired funtion. In the config file, there is a parameter called 'function'.
    You need to make sure to update the 'function' parameter with the desired function: create_dataset, train or predict_all
    """    
    config_manager.download_config_from_gcs()
    config=config_manager.init_config()
    task = config.get('function')
    if task =='create_dataset':
        create_dataset(config)   
    # elif task =='predict_all':
    #     predict_all(config)
    elif task == 'train':
        train_all(config)
    else:
        raise NotImplemented(f"Function {task} not implemented")
