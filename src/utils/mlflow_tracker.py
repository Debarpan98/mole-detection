from mailbox import NotEmptyError
import warnings
warnings.filterwarnings("ignore")
import mlflow
import datetime
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import pickle
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient
import torch

class MlFlowTracker():
    def __init__(self, model_name):

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            mlflow.log_param('Model', model_name)
        self.model_name = model_name

    def log_pd_file(self, df, name, index=False, folder='files'):
        """Save panda dataframe into a csv file and log the csv file using mlflow.
        Once the file is saved in mlflow, the file is deleted. 
        """
        df.to_csv(name, index=index )
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(name, folder)
        os.remove(name)

    def log_csv_file(self, csv_file):
        """Save panda dataframe into a csv file and log the csv file using mlflow.
        Once the file is saved in mlflow, the file is deleted. 
        """
        #np.savetxt(name,  file, delimiter=",")
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(csv_file, 'files')
    
    def log_pickle_file(self, dict, name):
        """Save dictionary into pickle file using mlflow.
        Once the file is saved in mlflow, the file is deleted. 
        """
        name = f"{self.model_name}-{name}"
        file_open = open(name, 'wb')
        pickle.dump(dict, file_open)
        
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(name, 'files')
        os.remove(name)

    
    def log_json_file(self, dictionnary, name):
        with open(name, 'w') as fp:
            json.dump(dictionnary, fp)
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(name, 'files')
        os.remove(name)

    def log_figure(self, figure: plt.figure, name:str):
        if figure is not None:
            figure.savefig(name,  bbox_inches = 'tight')
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_artifact( name, 'plots')
            os.remove(name)
        
    def log_params(self, params: dict):
        with mlflow.start_run(run_id=self.run_id):
            try:
                params.pop('params_settings', None)
            except:
                pass
            mlflow.log_params(params)
    
    def save_model(self, model, folder='models'):
#        if self.enable:
        model_output_dir = f'{self.model_name}.pth'
        torch.save(model.state_dict(), model_output_dir)
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(model_output_dir, folder)
        os.remove(model_output_dir)
            
    def log_param(self, param: object, name: str):
        
        with mlflow.start_run(run_id=self.run_id):
             mlflow.log_param(name, param)
    
    def log_metric(self, metric: object, name: str):        
        with mlflow.start_run(run_id=self.run_id):
             mlflow.log_metric( name,metric)
    
    def log_metrics(self, metrics: dict):           
        with mlflow.start_run(run_id=self.run_id):
             mlflow.log_metrics(metrics)
             
    def log_class_report(self, df, mode):            
        with mlflow.start_run(run_id=self.run_id):
            for index, row in df.iterrows():
                if index=='accuracy' or index=='subset_accuracy' or index=='hamming_score' or index=='hamming_loss' or index=='auc_score':
                    try:
                        mlflow.log_metric(f'{mode}-{index}', row['support'])
                    except:
                        pass
                else:
                    for column in df.columns:
                        try:
                            mlflow.log_metric(f'{mode}-{index}-{column}', row[column])
                        except:
                            pass
    

def log_avg_cv_metrics(experiment, run_ids, mlflow_tracker):
    df_metrics = mlflow.search_runs(experiment_ids=experiment.experiment_id)
    df_metrics = df_metrics[df_metrics['run_id'].isin(run_ids)].filter(like='metrics')
    df_metrics.columns = [col.replace('metrics.', "") for col in df_metrics.columns]
    mean_dict = {'AVG-' + str(key): val for key, val in dict(df_metrics.mean()).items()}
    std_dict = {'STD-' + str(key): val for key, val in dict(df_metrics.std()).items()}
    mlflow_tracker.log_metrics(mean_dict )
    mlflow_tracker.log_metrics(std_dict )

def update_ensemble_df(df, df_tmp):
    if df is None:
        df =  df_tmp
    else:
        if 'label' in df.columns:
            df = df.merge(df_tmp, on=['src', 'label'], how='outer')
        else:
            df = df.merge(df_tmp, on='src', how='outer')
    return df


    
