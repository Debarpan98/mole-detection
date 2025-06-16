from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from src.utils.utils import get_top_k_classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import PredefinedSplit
import src.utils.utils as utils
import torch.nn as nn
import torch
import numpy as np
import sklearn
import pandas as pd

class Results(ABC):
    """
    Factory for computing the results depending on the task at hand (multi-classification or multi-label)
    """

    @abstractmethod
    def get_predictions(self, outputs):
        pass
    
    @abstractmethod
    def get_probabilities(self,  outputs):
        pass
    
    @abstractmethod
    def get_predictions_names(self,  preds, label_encoder):
        pass

    @abstractmethod
    def get_df_results(self, probs, imgs, labels, pred_classes, label_encoder):
        pass

    @abstractmethod
    def get_confusion_matrix(self, original,predicted, label_encoder):
        pass

    @abstractmethod
    def get_class_report(self, original, predicted, target_names):
        pass

    @abstractmethod
    def get_df_predictions_for_ensemble(self, model_name: str, mode: str, probs: list, imgs: list, labels: list,
                                         label_encoder: sklearn.preprocessing)->pd.DataFrame:
        pass
    
    
    def get_columns_names(self, mode: str, model_name: str):
        columns= []
        if mode=='test':
            columns.append(f'{model_name}label')
            columns.append(f'{model_name}Pred1')
            columns.append( f'{model_name}Prob1')
            columns.append( f'{model_name}Pred2')
            columns.append(f'{model_name}Prob2')
            columns.append(f'{model_name}Pred3')
            columns.append(f'{model_name}Prob3')
        else:
            columns.append(f"{model_name}_label")
            columns.append( f"{model_name}_prediction")
            columns.append(f"{model_name}_proba")
            columns.append( f"{model_name}_y_pred1")
            columns.append(f"{model_name}_y_prob1")
            columns.append(f"{model_name}_y_pred2")
            columns.append(f"{model_name}_y_prob2")
        

        return columns


    def update_results(self, all_results, outputs, labels):
        all_results['all_preds'].extend(self.get_predictions(outputs).tolist())
        all_results['all_probs'].extend(self.get_probabilities(outputs).tolist())
        all_results['all_labels'].extend(labels.tolist())


    def get_top_1_2_3_classification_reports(self, df_preds:pd.DataFrame,  classes:list)-> dict:
        """return dictionary of top-1, top-2 and top-3 classification reports

        Args:
            df_preds (pd.DataFrame): predictions
            classes (list): list of classes

        Returns:
            dict:  top-1, top-2 and top-3 classification reports
        """    
        reports = {}
        for i in range(3):
            try:
                reports[i]= self.get_top_k_classification_report(df_preds, i+1, classes)
            except: 
                print(f' Warning - not able to generate top {i+1} report')
        return reports

    def get_top_k_classification_report(self, df_preds: pd.DataFrame, k: int, classes: list):
        """Generate top-k classification report

        Args:
            df_preds (pd.DataFrame): predictions
            classes (list): list of classes

        Returns:
            dict: top-k classification report
        """        
        return get_top_k_classification_report(df_preds, k, classes)


class ResultsMultiLabel(Results):
    def __init__(self, threshold, label_encoder):
        Results.__init__(self)
        self.threshold = threshold
        self.label_encoder = label_encoder

    def get_predictions(self, outputs):
        if torch.is_tensor(outputs):
            prob = torch.sigmoid(outputs)
            preds= prob.detach().cpu().apply_(lambda x: 1 if x>self.threshold else 0)
        else:
            threshold_vectorized= np.vectorize((lambda x: 1 if x>self.threshold else 0))
            preds = threshold_vectorized(outputs)
        return preds
    
    def get_probabilities(self,outputs):
        return  torch.sigmoid(outputs)

    def get_predictions_names(self,  preds, label_encoder): 
        probs_classes = np.empty([preds.shape[0]], dtype=object)
        if torch.is_tensor(preds):
            preds_list = preds.nonzero()
        else:
            preds_list = np.transpose(preds.nonzero())
        for pred in preds_list:
            idx = pred[0].item()
            value = label_encoder.inverse_transform([pred[1].item()]).item()
            if probs_classes[idx] is None:
                probs_classes[idx] = [value]
            else:
                probs_classes[idx].append(value)
        return  list(probs_classes)

    def get_df_results(self, probs: list, imgs: list, labels: list, preds:list, pred_classes: list,  label_encoder: sklearn.preprocessing)->pd.DataFrame:
        """Created pandas dataframe that contains all results (labels, predicted values ...)
        Args:
            probs (list): list of output probabilites for predicted class
            imgs (list): list of path to images
            labels (list): list of true labels
            pred_classes (list): list of predicted values
            class_names_list (dict): dictionary that maps class num with class label

        Returns:
            pandas.DataFrame: dataframe that contains all results 
        """    
        frame = pd.DataFrame(probs)
        frame.columns = ["prob_{}".format(d) for d in  label_encoder.classes_]
        frame["filename"] = imgs
        frame["label_names"] = self.get_predictions_names(np.array(labels), self.label_encoder)
        frame["predicted_names"] = pred_classes
        frame['label']= labels
        frame['preds'] =preds.tolist()

        return frame 


    def get_confusion_matrix(self, original,predicted, label_encoder):
        return utils.get_confusion_matrix(original, predicted, label_encoder, is_multi_label=True)

    def get_class_report(self, original, predicted, probs, target_names):
        report = classification_report(original, predicted, target_names = target_names, output_dict=True)
        report['subset_accuracy'] = sklearn.metrics.accuracy_score(original, predicted, normalize=True, sample_weight=None)
        report['hamming_score'] = utils.hamming_score(np.array(original), np.array(predicted))
        report['hamming_loss'] = sklearn.metrics.hamming_loss(original, predicted)
        try:
            report['auc_score'] = roc_auc_score(original, probs)
        except:
            report['auc_score'] =None
        return pd.DataFrame(report).transpose()

    def compute_threshold(self, labels, probs):
        _, _, thresholds  = roc_curve(np.array(labels).ravel(), np.array(probs).ravel())
        idx = int(len(thresholds)/2)
        self.threshold =  thresholds[idx]
        print(f'Custom threshold: {self.threshold:0.2}')

    def get_df_predictions_for_ensemble(self, model_name: str, mode: str, probs: list, imgs: list, labels: list,
                                         label_encoder: sklearn.preprocessing)->pd.DataFrame:
        """Created pandas dataframe that contains all results (labels, predicted values ...)
        Args:
            probs (list): list of output probabilites for predicted class
            imgs (list): list of path to images
            labels (list): list of true labels
            pred_classes (list): list of predicted values
            class_names_list (dict): dictionary that maps class num with class label

        Returns:
            pandas.DataFrame: dataframe that contains all results 
        """    
        frame= pd.DataFrame(imgs, columns=['src'])
        #frame["true_label"] = self.get_predictions_names(np.array(labels), self.label_encoder)
        columns = self.get_columns_names(mode, model_name)
        frame[columns[0]] = [item[0] if len(item) == 1 else item for item in np.array(self.get_predictions_names(np.array(labels), self.label_encoder))]
        frame[columns[1]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 0])
        frame[columns[2]] = np.flip(np.sort(probs), axis=1)[:, 0]
        frame[columns[3]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 1])
        frame[columns[4]] = np.flip(np.sort(probs), axis=1)[:, 1]
        frame[columns[5]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 2])
        frame[columns[6]] = np.flip(np.sort(probs), axis=1)[:, 2]

        frame2 = pd.DataFrame(probs)
        frame2.columns = ["{}_prob_{}".format(model_name,d) for d in  label_encoder.classes_]
        frame2['label'] =  [item[0] if len(item) == 1 else item for item in np.array(self.get_predictions_names(np.array(labels), self.label_encoder))]
        frame2['src']= imgs
        frames = {'1': frame, '2': frame2}
        return frames 


class ResultsMultiClass(Results):
    def __init__(self, label_encoder):
        Results.__init__(self)
        self.label_encoder = label_encoder

    def get_predictions(self, outputs):
        if torch.is_tensor(outputs):
            _, preds = torch.max(outputs.data, 1)
        else:
            preds = np.argmax(softmax(np.array(outputs), axis=1), axis=1)
        return preds
    
    def get_probabilities(self, outputs):
        return  torch.stack([nn.Softmax(dim=0)(i) for i in outputs])  

    def get_predictions_names(self,  preds, label_encoder): 
        return label_encoder.inverse_transform(preds)
    

    def get_df_results(self, probs: list, imgs: list, labels: list, preds:list, pred_classes: list,  label_encoder: sklearn.preprocessing)->pd.DataFrame:
        """Created pandas dataframe that contains all results (labels, predicted values ...)
        Args:
            probs (list): list of output probabilites for predicted class
            imgs (list): list of path to images
            labels (list): list of true labels
            pred_classes (list): list of predicted values
            class_names_list (dict): dictionary that maps class num with class label

        Returns:
            pandas.DataFrame: dataframe that contains all results 
        """    
        frame = pd.DataFrame(probs)
        frame.columns = ["prob_{}".format(d) for d in  label_encoder.classes_]
        frame["filename"] = imgs
        frame["label"] = label_encoder.inverse_transform(labels)
        frame["predicted"] = pred_classes
        frame['preds'] = preds
        return frame 

    def get_df_predictions_for_ensemble(self, model_name: str, mode:str, probs: list, imgs: list, labels: list, 
                                       label_encoder: sklearn.preprocessing)->pd.DataFrame:
        """Created pandas dataframe that contains all results (labels, predicted values ...)
        Args:
            probs (list): list of output probabilites for predicted class
            imgs (list): list of path to images
            labels (list): list of true labels
            pred_classes (list): list of predicted values
            class_names_list (dict): dictionary that maps class num with class label

        Returns:
            pandas.DataFrame: dataframe that contains all results 
        """    
        frame= pd.DataFrame(imgs, columns=['src'])
        columns = self.get_columns_names(mode, model_name)
        frame[columns[0]] = self.get_predictions_names(np.array(labels), self.label_encoder)
        frame[columns[1]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 0])
        frame[columns[2]] = np.flip(np.sort(probs), axis=1)[:, 0]
        frame[columns[3]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 1])
        frame[columns[4]] = np.flip(np.sort(probs), axis=1)[:, 1]
        # frame[columns[5]] = self.label_encoder.inverse_transform(np.flip(np.argsort(probs), axis=1)[:, 2])
        # frame[columns[6]] = np.flip(np.sort(probs), axis=1)[:, 2]

        frame2 = pd.DataFrame(probs)
        frame2.columns = ["{}_prob_{}".format(model_name,d) for d in  label_encoder.classes_]
        frame2['src']= imgs
        frame2['label'] = self.get_predictions_names(np.array(labels), self.label_encoder)
        frames = {'1': frame, '2': frame2}
        return frames 

    
    def get_confusion_matrix(self, original,predicted, label_encoder):
        return utils.get_confusion_matrix(original,predicted, label_encoder, is_multi_label=False)

    def get_class_report(self, original, predicted, probs, target_names):
        report = pd.DataFrame( classification_report(original, predicted, target_names=target_names, output_dict=True)).transpose()
        return report


class ResultsFactory:
    def create_task(self, multi_label, threshold=None, label_encoder=None):
        if  multi_label:
            return ResultsMultiLabel(threshold, label_encoder)

        else:
            return ResultsMultiClass(label_encoder)


