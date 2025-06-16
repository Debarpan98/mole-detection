from matplotlib.cbook import is_math_text
from src.utils.utils import visualize_loss, save_pdf_general, visualize_learning_rates
from src.utils.mlflow_tracker import MlFlowTracker
from sklearn.metrics import roc_auc_score,accuracy_score
import pandas as pd


class TrainingMetrics():
     """ Class to gather all the training metrics 
        including training losses, training accuracies, 
        validation losses, validation accuracies 
        and the learning rate values --> over the epochs
    """    
     def __init__(self, is_multi_label):
        """Initialization

        Args:
            predicitons (pandas.DataFrame): dataframe that contains all results 
            class_report (dict): classification report
            confusion_matrix (matplotlib.pyplot.figure): confusion matrix
        """        
        self.training_losses = []
        self.validation_losses = []
        self.training_scores = []
        self.validation_scores = []
        self.learning_rates = []
        self.loss_f, self.scores_f, self.learning_rates_f = None, None, None
        self.is_multi_label = is_multi_label
        if is_multi_label:
            self.score_str = 'AUC'
        else:
            self.score_str= 'ACC'
        
     def get_losses_per_mode(self, mode: str)->list:
         """Returns losses per mode (over epochs)

         Args:
             mode (str): mode mode ['train', 'val', 'test']

         Returns:
             list: list of losses
         """         
         if mode =='train':
            return self.training_losses
         else:
            return self.validation_losses
            
     def get_scores_per_mode(self,mode: str)->list:
         """Returns accuracies per mode (over epochs)

         Args:
             mode (str): mode mode ['train', 'val', 'test']

         Returns:
             list: list of accuracies
         """         
         if mode =='train':
            return self.training_scores
         else:
            return self.validation_scores

     def add_scores(self, mode: str, all_labels:list, all_preds:list, all_probs:list):
        scores = self.get_scores_per_mode(mode)
        if self.is_multi_label is False:
            score = accuracy_score(all_labels, all_preds)
            printed_metric = f'ACC: {score:.4f}'
        else:
            try:
                score = roc_auc_score(all_labels, all_probs)
                printed_metric = f'AUC: {score:.4f}'
            except:
                score = None
                printed_metric = None
        scores.append(score)
        return printed_metric
        
        
     def generate_plots(self):
         """Generates plots - losses over epochs, accuracies or AUC scores over epochs and learnign rate over steps
         """         
         self.loss_f = visualize_loss(self.training_losses, self.validation_losses, 'Loss')
         self.scores_f = visualize_loss(self.training_scores, self.validation_scores, f'{self.score_str} (%)')
         self.learning_rates_f = visualize_learning_rates(self.learning_rates)
         
     def save_metrics(self, folder_datetime:str)->None:
         """Save metrics in pdf files

         Args:
             folder_datetime (str): target folder path
         """    
         if self.loss_f is None:
             self.generate_plots()
         save_pdf_general(self.loss_f, folder_datetime  / f"losses_over_epochs.pdf")
         save_pdf_general(self.scores_f, folder_datetime  / f"acc_over_epochs.pdf")
         save_pdf_general(self.learning_rates_f, folder_datetime  / f"lr_over_epochs.pdf" )
         
         
     def save_metrics_mlflow(self, mlflow_tracker: MlFlowTracker, model_name)->None:
         """Save metrics in mlflow

         Args:
             mlflow_tracker
         """    
         if self.loss_f is None:
              self.generate_plots()
         mlflow_tracker.log_figure(self.loss_f, name=f"{model_name}-losses_over_epochs.pdf")
         mlflow_tracker.log_figure(self.scores_f, name=f"{model_name}-acc_over_epochs.pdf")
         mlflow_tracker.log_figure(self.learning_rates_f, name=f"{model_name}-lr_over_epochs.pdf")

