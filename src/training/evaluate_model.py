import torch
import torch.nn as nn
import numpy as np
import sklearn
from src.utils.mlflow_tracker import MlFlowTracker
from src.utils.utils import  plot_auc_curve 
from torch.autograd import Variable
from src.training.compute_results import ResultsFactory, ResultsMultiLabel


class Results():
    """ Class that gather all the experiment results 
    """ 
    def gather_all_results(self,model_name:str, mode:str,  original: list, probs:list, imgs: list, results:ResultsFactory, 
                          label_encoder: sklearn.preprocessing, dataloader: torch.utils.data.DataLoader):
        
        predicted = results.get_predictions(probs)
        probs_classes = results.get_predictions_names(predicted, label_encoder)
        self.class_report = results.get_class_report(original, predicted, probs, target_names=label_encoder.classes_)
        self.frame = results.get_df_results(probs, imgs, original, predicted , probs_classes, label_encoder )  
        self.ensemble_frames = results.get_df_predictions_for_ensemble(model_name, mode, probs, imgs, original, label_encoder)
        self.top_1_2_3_reports = results.get_top_1_2_3_classification_reports(self.frame , classes=label_encoder.classes_)  
        self.confusion_matrix = results.get_confusion_matrix(original,predicted, label_encoder)
        
        #num_classes = len(label_encoder.classes_)
        
        # if num_classes == 2:
        #     # binary classification
        #     self.misclassified_acne_plot =  plot_binary_misclassified_bar_chart(self.frame, dataloader.dataset.df_dataset, misclassified_label='acne')
        #     self.misclassified_not_acne_plot =  plot_binary_misclassified_bar_chart(self.frame, dataloader.dataset.df_dataset,  misclassified_label='not_acne')
        # else:
        #     self.misclassified_acne_plot =None
        #     self.misclassified_not_acne_plot= None
        try:
            self.roc_curve =  plot_auc_curve (np.array(original),np.array(probs), label_encoder.classes_)
        except:
            self.roc_curve = None 
        

def model_evaluation(dataloader: torch.utils.data.DataLoader, mode: str, model_conv: nn.Module, model_name: str, 
                    label_encoder: sklearn.preprocessing, results:ResultsFactory)->Results:
    
    """ Function that evaluate performance of a model on a image set

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader
        model_conv (nn.Module): model
        label_encoder (sklearn.preprocessing): label encoder
        results (ResultsFactory): object to gather all the results


    Returns:
        Results: object that contains all the evaluation results
    """    
    original,  probs, imgs = [], [], []
    model_conv.eval()
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    with torch.no_grad():
        for elem in dataloader:
            inputs, labels, image_names = elem
            imgs.extend(image_names)
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model_conv(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            prob = results.get_probabilities(outputs)
            probs.extend(prob.cpu().numpy())
            original.extend(labels.cpu().numpy())

    if isinstance(results, ResultsMultiLabel):
        results.compute_threshold(original, probs)
    
    all_results = Results()
    all_results.gather_all_results(model_name, mode,  original, probs, imgs,results, label_encoder, dataloader)
    return all_results

def save_results(mode: str, mlflow_tracker: MlFlowTracker,  model_name: str, metrics:Results) -> None:
    """Saves predictions dataframe and classification report to google bucket
        or local file
    Args:
        mode (str): mode ['train', 'val', 'test']
        class_report (dict): classification report
        predictions (pandas.DataFrame): dataframe that contains all the experiment results
        confusion_matrix (plt.figure) : confusion_matrix figure
    """    
    if isinstance(metrics, ResultsMultiLabel):
        mlflow_tracker.log_param(metrics.threshold, f'{mode}-multi-label-threshold')
   
    if mode =='test':
        mlflow_tracker.log_class_report(metrics.class_report, mode)
    #    for report_key in metrics.top_1_2_3_reports.keys():
            # we also want to log the top-3 predictions
    #        if report_key==2:
    #            mlflow_tracker.log_class_report(metrics.top_1_2_3_reports[report_key], 'top-3')

    for report_key in metrics.top_1_2_3_reports.keys():
        mlflow_tracker.log_pd_file(metrics.top_1_2_3_reports[report_key],  f"{model_name}-{mode}_top_{report_key +1}_class_report.csv", index=True)

    mlflow_tracker.log_pd_file(metrics.frame,  f"{model_name}-{mode}_predictions.csv")
    mlflow_tracker.log_figure(metrics.confusion_matrix, f"{model_name}-{mode}_confusion_matrix.pdf")

    if metrics.roc_curve is not None:
        mlflow_tracker.log_figure(metrics.roc_curve, f"{model_name}-{mode}_roc_curve.pdf")
   
    #mlflow_tracker.log_pd_file(metrics.class_report,  f"{model_name}-{mode}_class_report.csv", index=True)
    # if metrics.misclassified_bar_charts is not None:
    #     mlflow_tracker.log_figure(metrics.misclassified_acne_plot, f"{model_name}-{mode}_misclassidied_acne_images.pdf")
    #     mlflow_tracker.log_figure(metrics.misclassified_not_acne_plot, f"{model_name}-{mode}_misclassidied_not_acne_images.pdf")

def evaluate_model(model: nn.Module, model_name:str, mlflow_tracker: MlFlowTracker, dataloaders: dict,  
                  label_encoder: sklearn.preprocessing, results:ResultsFactory) -> None:
    """Function that evaluate the performance of the trained model on the train, val and test set.
    This function is also called to get the predictions on the entire dataset. 

    Args:
        model (nn.Module): model
        model_name (str): name of the model
        mlflow_tracker (MlflowTracker): mlflow tracker
        dataloaders (dict): dictionary of dataloaders (train, val and test)
        class_indices (dict): dictionary that maps class numbers to class labels
    """
    if isinstance(dataloaders, dict):
        for mode in dataloaders.keys():
            print("[{}- Evaluating the data for mode: {}]".format(model_name, mode))
            metrics = model_evaluation(dataloaders[mode], mode, model, model_name, label_encoder, results)
            save_results(mode, mlflow_tracker, model_name, metrics)

    else:
        print('Model evaluation starting.')
        metrics = model_evaluation(dataloaders, 'test',  model, model_name,  label_encoder, results)
        save_results('inference', mlflow_tracker, model_name, metrics)
        print('Model evaluation completed.')


