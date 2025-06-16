"""
Contains Configuration Manager Class and functions needed to use it
"""
from envyaml import EnvYAML
from google.cloud import storage

from src.utils.utils import check_file_exists
from pathlib import Path
import src.utils.envs  as envs
import os
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

# GLOBAL VARIABLES
e=envs.Envs()
config=None

class ConfigManager:
    """Configuration Manager Class.
        The class attribute 'config' is a dictionary
        that contains all the settings and hyperparameters
        necessary to conduct the experiment. 
    """
    def __init__(self, config):
        self.config = config

    def get(self,key):
        return self.config[key]
    
    def set(self, key,new_val):
        self.config[key]=new_val
    
    def get_model_param(self, model_name, param):
        """Returns the hyperparameter named 'param'
            for model named 'model_name'

        Args:
            model_name (str): model name
            param (str): model parameter/hyperparameter

        Returns:
            str: value of parameter/hyperparameter
        """    
        return self.config.get("model")[model_name][param]
    
    def set_model_param(self, model_name, param, new_value):
        """Set the hyperparameter named 'param'
            for model named 'model_name'

        Args:
            model_name (str): model name
            param (str): model parameter/hyperparameter
            new_value (object) : new value

        Returns:
            str: value of parameter/hyperparameter
        """    
        self.config.get("model")[model_name][param] = new_value

def app_config(config_path='../configs/sdd_config.yaml'):

    while True:
        try:
            config = EnvYAML(config_path)
            return config['sdd_config_values']
        
        except:
            #print(f"Error readining config file:  {config_path} ")
            pass
 
def download_config_from_gcs():
    """This function downloads the config file from a
        Google Cloud bucket and save it as "config_recent.yaml".
    """
    if check_file_exists("config_recent.yaml"):
        try:
            os.remove("config_recent.yaml")
        except:
            print("config_recent does not exist")

    storage_client = storage.Client()
    bucket = storage_client.bucket(e.gcs_bucket)
#    blob = bucket.blob('sdd_Debarpan/configs/sdd_files_configs_vm_test.yaml')
    blob = bucket.blob('sdd_Debarpan/configs/sdd_files_configs_vm.yaml')
    blob.download_as_string()
    with open("config_recent.yaml", "wb") as file_obj:
        blob.download_to_file(file_obj)
    return True

def init_config():
    global config

    if(config==None):
        print('Using config file from environment variable GCS')
        config=app_config('config_recent.yaml')

        ##Overriding root for further usage
        config["root"] = 'gs://' + config.get('gcs_bucket') + '/' + config.get('gcs_folder') + '/'
        config["gcs_bucket_img"]=e.gcs_image_bucket
        config["gcs_bucket"] =e.gcs_bucket

    return ConfigManager(config)