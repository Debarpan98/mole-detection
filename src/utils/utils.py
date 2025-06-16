""" Helpers for the main script

Functions:
   logger_config()    -- logger configuration

"""
import sys
import cv2
import pickle
import torch
import numpy as np
import sklearn
import random
import os
import google.auth
import math
import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import gcsfs
import pandas as pd
from src.utils.gcs_utils import get_bucket_gcs, save_from_filename
from src.data.skin_functions import calculate_skin_percentage_general_new
from sklearn.metrics import roc_curve, auc
from joblib import load
from io import StringIO
from src.utils import envs 
from sklearn.metrics import classification_report
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import multilabel_confusion_matrix

all_classes = ['atopic_dermatitis','psoriasis_vulgar','tinea_corporis','granuloma_annulare','peri_oral_dermatitis','shingles','dyshidrosis','rosacea_erythemato_telangiectasique','tinea_versicolor']


e=envs.Envs()

def logger_config():
    return {
        "handlers": [
            {"sink": sys.stdout, "format": "{time} - {message}", "level": "DEBUG"},
            {"sink": "ds-sdd.log", "serialize": True, "level": "DEBUG","rotation":"1 week"},
        ]
    }


def pickle_dump_gen(pickle_file,pickle_file_path, gc_project, gcs_folder= None):
    """Created pickle file and save it 

    Args:
        pickle_file (object): data to save
        pickle_file_path (str): path to file

        gc_project (str): google cloud project
        gcs_folder (str) : google cloud folder
        config (dict): dictionnary of config parameters
        is_gcs (bool, optional): indicating if GCS. Defaults to False.
    """
    credentials,_ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    fs = gcsfs.GCSFileSystem(project=gc_project, token=credentials)
    fs.ls(e.gcs_bucket)
    with fs.open(e.gcs_bucket+'/' + gcs_folder + '/' +pickle_file_path, 'wb') as handle:
        pickle.dump(pickle_file, handle)


def resize(img, height,width):
    """Resize image to desired size

    Args:
        img: image that need to be resized
        height (int): desired height
        width (int): desired width

    Returns:
        [type]: [description]
    """

    img = cv2.resize(img, ((height,width)), interpolation=cv2.INTER_AREA)
    return img


def setAllSeeds(seed):
    """Helper for setting seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def class_dictionary_create(class_list =None):
    """Get dictionnary that maps the name of the classes
       and with the classes number

    Args:
        file_dir (str): path to file 
        img_src_type (int): type of data source
        class_list (list, optional): list of the classes. Defaults to None.

    Returns:
        dict: dictionnary mapping class names and class numbers
    """
    labels= class_list
    values=pd.factorize(labels)[0]
    class_dictionary = {labels[i]: values[i] for i in range(len(labels))}
    return class_dictionary


def check_folder_exists(path):
    return os.makedirs(path, exist_ok=True)

def check_file_exists(path):
    return os.path.isfile(path)


def joblib_load_gen(file_path, config):
    file_bucket = config.get('gcs_folder') + '/' + file_path
    bucket = get_bucket_gcs(config.get('gcs_bucket'))
    blob = bucket.blob(file_bucket)
    blob.download_to_filename(config.get('root_vm') + file_path)
    file_load = load(config.get('root_vm') + file_path)
    return file_load


def save_pdf_general(f, save_pathname):
    """Save plot in .PDF format

    Args:
        f (matplotlib.pyplot.figure): figure
        save_pathname ([type]): [description]
    """    
    folder_time = str(datetime.datetime.now())
    model_output_dir = str(folder_time + Path(save_pathname).name)
    f.savefig(model_output_dir)
    save_from_filename(e.gcs_bucket, str(save_pathname), model_output_dir)
    os.remove(model_output_dir)

def save_csv_general(csv_file, save_pathname):
    f = StringIO()
    try:
        csv_file.to_csv(f,  encoding="utf-8")
        f.seek(0)
        get_bucket_gcs(e.gcs_bucket).blob(str(save_pathname)).upload_from_file(f,content_type='text/csv')
        print('Dataset saved in utf-8')
    except:
        csv_file.to_csv(f,  encoding="latin-1")
        f.seek(0)
        get_bucket_gcs(e.gcs_bucket).blob(str(save_pathname)).upload_from_file(f,content_type='text/csv')
        print('Dataset saved in latin-1')
    # "UnicodeEncodeError: 'latin-1' codec can't encode character '\u0300' in position 2356362: Body ('̀') is not valid Latin

def visualize_loss(list_train: list, list_test:list, y_label: str)-> plt.figure:
    """
    Visualizes train and test values
    Args:
        list_train (list): list of train values
        list_test (list): list of test values
        y_label (str): y label 
    Returns:
        f (plt.figure): figure of losses over epocs
    """ 
    f = plt.figure(figsize=(7,5))
    epochs = np.arange(0, len(list_train))
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.plot( epochs, list_train, label='Train' )
    plt.plot( epochs, list_test, label= 'Val' )
    plt.legend()
    return f 

def visualize_learning_rates(learnin_rates: list) -> plt.figure:  
    """Created plot with learnign rates over iterations

    Returns:
        plt.figure: figure
    """
    f = plt.figure(figsize=(7,5))
    epochs = np.arange(0, len(learnin_rates))
    plt.xlabel('Steps')
    plt.ylabel('LR')
    plt.plot( epochs, learnin_rates)
    return f 


def compute_roc_curve(y_test:list, y_score:list, n_classes:int):
    """Compute auc score, false positive and true positive

    Args:
        y_test (list): list of labels
        y_score (list): list of probabilities
        n_classes (int): number of classes
    """    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc, fpr, tpr

def plot_auc_curve(y_test: list, y_score:list, classes:list) -> plt.figure:
    """Generated ROC curve

    Args:
        y_test (list): list of labels
        y_score (list): list of probabilities
        classes (list): list of diseases

    Returns:
        plt.figure: roc curve
    """  
    roc_auc, fpr, tpr = compute_roc_curve(y_test, y_score, len(classes))
    f =plt.figure(figsize=(10,10))
    lw = 2
    for i , disease in enumerate(classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label=f"ROC curve {disease} (area = {roc_auc[i]:.2f})"
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    return f

def compute_accuracy_per_disease(matrix: np.array, classes: list) -> list:
    """Multi-label regime: compute the accuracy for each disease (one vs all)

    Args:
        matrix (np.array): multi-label confusion matrix
        classes (list): list of diseases 

    Returns:
        (list): list of accuracies for each disease
    """ 
    accs = []
    for i, disease in enumerate(classes):
        sub_matrix = matrix[i]
        total = np.sum(sub_matrix)
        diag = np.sum(np.diagonal(sub_matrix))
        accs.append(diag/total)
    return accs

def print_confusion_matrix(confusion_matrix: np.array, axes: plt.axes, class_label: str, class_names: list, accuracy: int, fontsize: int=30):
    """
    Print confusion matrix -> this function is only called to show the confusion matrix for multi-label task (one vs all)
    Adapated FROM: https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    Args:
        confusion_matrix (np.array): confusion matrix
        axes (plt.axes): figure axes
        class_label (str): class labels
        class_names (list): columns to display on the confusion matrix -> for example ["F", "T"]
        accuracy (int): accuracy
        fontsize (int, optional) font size. Defaults to 30.
    """

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, annot_kws={"size": fontsize})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label', fontsize=fontsize)
    axes.set_xlabel('Predicted label', fontsize=fontsize)
    axes.set_title(f"{class_label}: {accuracy:.2f}%", fontsize=fontsize )        

def get_confusion_matrix(y_true: list, y_pred:list, label_encoder: sklearn.preprocessing, is_multi_label:bool=False,
                         col:int=5, fontsize:int=30) -> plt.figure:
    """
   Get confusion matrix - figure of confusion matrix
    Args:
        y_true (list): list of labels
        y_pred (list): list of predicted values
        label_encoder (sklearn.preprocessing): label encoder
        is_multi_label (bool): bool indicating if we are in the mutli label regime. Defaults False. 
        col (int): number of columns. Defaults 5.
        fontsize (int). Font size. Defaults 30.
    
    Returns:
        f (matplotlib.pyplot.figure): figure
    """    
    if is_multi_label:
        matrix = multilabel_confusion_matrix(y_true, y_pred)
        accs = compute_accuracy_per_disease(matrix, label_encoder.classes_)
        num_classes = len(label_encoder.classes_)
        row= math.ceil(num_classes / col)
        fig_h = int(7 * row)
        
        f, ax = plt.subplots(row, col, figsize=(70,  fig_h))
        plt.title('Multi-Label Confusion Matrices with threshold of {threshold}.', fontsize=fontsize)
        for axes, cfs_matrix, label, acc in zip(ax.flatten(), matrix , label_encoder.classes_, accs):
            print_confusion_matrix(cfs_matrix, axes, label, ["F", "T"], acc,  fontsize=fontsize )
        plt.subplots_adjust(left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.4, 
            hspace=0.4)
    else:
        matrix = confusion_matrix(y_true, y_pred)
        index = label_encoder.classes_
        index_renamed = []
        for i in index:
            index_renamed.append(i.replace('acne_',''))
        df_cm = pd.DataFrame(matrix, index=index_renamed, columns=index_renamed)
        f = plt.figure(figsize=(20,  20))
        sn.heatmap(df_cm, annot=True,  fmt='g')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    return f

def hamming_score(y_true: np.array, y_pred: np.array)->int:
    """ Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
        https://stackoverflow.com/q/32239577/395857
        FROM: https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit

    Args:
        y_true (np.array): list of labels
        y_pred (np.array): list of predictions

    Returns:
        int: hamming score
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def get_hyperparams(params: dict):
    """Get the list of hyperparameters 
    Args:
        params (dict): dictionary of hyperparameters 

    Returns:
        lsit: complete list of combination of hyperparameters
    """    
    # Define the grid search for the hyperparameters
    hparam_grid = {}
    for key in params.keys():
        hparam_grid[key] = params[key]

    return list(ParameterGrid(hparam_grid))

def set_hyperparams(model_name:str, config:dict, hparam_dict:dict): 
    """Set new hyperparameters in the config file the from hparam_dict dictionary. 

    Args:
        model_name (str): model name
        config (dict): dictionary of hyperparameters and settings
        hparam_dict (dict): dictionary of new hyperparameters to update
    """
    for key in hparam_dict:
        config.set_model_param( model_name, key, hparam_dict[key])


def sample_per_disease(column: str, df: pd.DataFrame, num_images: int, max_value: int, subsample_max_per_disease: dict)-> None:
    """Sample dataset per disease. The disease is 'column'. 
    We want to only keep a maimum of max_value images. However, if the disease is in the subsample_max_per_disease dictionary, then 
    we keep the number specified in the dictionary. 
    Args:
        column (str): disease
        df (pd.DataFrame): dataset
        num_images (int): number of images for the disease in the dataset
        max_value (int): maximum number of images 
        subsample_max_per_disease (dict): dictionary of diseases and max values
    """  
    if column in subsample_max_per_disease.keys():
        max_value = subsample_max_per_disease[column]
    if num_images > max_value:
        print(f'{column} has {num_images}, but max value is {max_value}.')
        idx= df[df[column]==True].sample(n= num_images - max_value,random_state=42).index
        df.drop(idx, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)


def increase_per_disease(column: str, df_dataset: pd.DataFrame, df_train: pd.DataFrame, num_images: int, max_value: int, subsample_max_per_disease: dict)-> None:
    """Sample dataset per disease. The disease is 'column'. 
    We want to only keep a maimum of max_value images. However, if the disease is in the subsample_max_per_disease dictionary, then 
    we keep the number specified in the dictionary. 
    Args:
        column (str): disease
        df (pd.DataFrame): dataset
        num_images (int): number of images for the disease in the dataset
        max_value (int): maximum number of images 
        subsample_max_per_disease (dict): dictionary of diseases and max values
    """  
    if column in subsample_max_per_disease.keys():
        max_value = subsample_max_per_disease[column]
    
    if num_images < max_value:
        print(f'{column} has {num_images}, but max value is {max_value}.')
        num_images = max_value - num_images
        more_images = df_dataset[df_dataset[column]==True].shape[0]
        if more_images < num_images:
            num_images = more_images
        idx= df_dataset[df_dataset[column]==True].sample(n= num_images ,random_state=42).index
        df_train = pd.concat([df_train, df_dataset.loc[idx, ]]).reset_index(drop=True)
    return df_train

def increase_with_max_value(df_dataset: pd.DataFrame, df_train: pd.DataFrame,  max_value:int, subsample_max_per_disease: dict)->pd.DataFrame:
    """Increase dataset per disease.
    Only keep 'max_value' images per disease.
    However, if a disease is in the 'subsample_max_per_disease' dictionary, then for this disease, 
    keep the number of images specified in the dictionary 
    --> for example, if the max_value is 100, and the subsample_max_per_disease={acne: 200, 'peri': 300}, then we keep a maximum of
    200 images for acne, 300 for peri and 100 for all the other diseases. 

    Args:
        df (pd.DataFrame): dataset 
        max_value (int): maximum value of images per diseases 
        subsample_max_per_disease (dict): dictionary where the keys are the disease names and the values are the maximum number of images

    Returns:
        df (pd.DataFrame): dataset 
    """
    for column in df_train.columns:
        try:
            if column != 'pathBucketImage' and column != 'diseases':
                num_images = df_train[column].value_counts()[True] 
                df_train =increase_per_disease(column, df_dataset, df_train, num_images, max_value, subsample_max_per_disease)
        except:
            pass
    print(f'Number of images after increase of images: {df_train.shape[0]}.')
    return df_train



def sample_with_max_value(df: pd.DataFrame,  max_value:int, subsample_max_per_disease: dict)->pd.DataFrame:
    """Sample dataset per disease.
    Only keep 'max_value' images per disease.
    However, if a disease is in the 'subsample_max_per_disease' dictionary, then for this disease, 
    keep the number of images specified in the dictionary 
    --> for example, if the max_value is 100, and the subsample_max_per_disease={acne: 200, 'peri': 300}, then we keep a maximum of
    200 images for acne, 300 for peri and 100 for all the other diseases. 

    Args:
        df (pd.DataFrame): dataset 
        max_value (int): maximum value of images per diseases 
        subsample_max_per_disease (dict): dictionary where the keys are the disease names and the values are the maximum number of images

    Returns:
        df (pd.DataFrame): dataset 
    """
    for column in df.columns:
        try:
            if column != 'pathBucketImage' and column != 'diseases':
                num_images = df[column].value_counts()[True] 
                sample_per_disease(column, df, num_images, max_value, subsample_max_per_disease)
        except:
            pass
    print(f'Number of images after subsampling: {df.shape[0]}.')
    return df

def remove_images_skin_ratio(ratio:int, df: pd.DataFrame)->pd.DataFrame:
    """Remove images from dataset with skin ratio > threshold

    Args:
        ratio (int): skin threshold
        df_final (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: updated dataset
    """    
    df = df.drop(df[df.ratios < ratio].index)
    print(f"Number of images after removing images with skin ratio < {ratio}: {df.shape[0]}")
    return df

def ratio_check_df(df:pd.DataFrame, size:list, skin_ratio_threshold:float =0.5):
    """Verify if the images contain enough skin 

    Args:
        df (dataframe): pandas dataframe of all the images
        size (list): image size (height, width)
        skin_ratio_threshold (float, optional): skin ratio threshold. Defaults to 0.5.

    Returns:
        df (dataframe): dataframe without the images <skin ratio threshold
    """
    for index, row in df.iterrows():
        image_read=None
        bucket = get_bucket_gcs(e.gcs_image_bucket)
        try:
            image_read = cv2.resize(cv2.imdecode(np.asarray(bytearray(bucket.blob(row['filename']).download_as_string()), dtype=np.uint8), cv2.IMREAD_COLOR),(size[0],size[1]))
        except:
            print("Image cannot be read during ratio_check_df")
            print(row['filename'])
        if image_read is None:
            df.drop(index, inplace=True)
        else:
            ratio = calculate_skin_percentage_general_new(image_read)
            if ratio<=skin_ratio_threshold:
                df.drop(index, inplace=True)
    return  df


def plot_binary_misclassified_bar_chart(df_preds: pd.DataFrame, df_dataset: pd.DataFrame,  misclassified_label: str ='acne', fig_size: int =7 ):    
    """Generates bar chart --> only for the binary classification task
       The figure shows the number of images that are misclassified per disease ( and the total number of images per disease.)

    Args:
        df_preds (pd.DataFrame): all predictions
        df_dataset (pd.DataFrame): dataset of all images
        misclassified_label (str, optional): misclassified class. Defaults to 'acne'.
        fig_size (int, optional): Figure Width . Defaults to 7.

    Returns:
        _type_: _description_
    """
    df_preds = df_preds.merge(df_dataset, how='left', on='filename')
    df_misclassify = df_preds[df_preds['label']!=df_preds['predicted']].reset_index(drop=True)
    
    if misclassified_label=='acne':
        label ='not_acne'
    else:
        label='acne'

    f = plt.figure(figsize=(23,fig_size))
    all_values= df_preds[df_preds['label']==label]['diseases'].value_counts().sort_index()
    values = df_misclassify[df_misclassify['predicted']==misclassified_label]['diseases'].value_counts()
    for idx in all_values.index:
        if idx not in values:
            values = values.append(pd.Series([0],[idx]))
    values= values[all_values.index]
    all_values.plot(kind='bar', label='all not acne images', color='orange')
    values.plot(kind='bar', label='misclassified not acne images', color='b')
    plt.ylabel('Num of Images')
    plt.xlabel('Not acne disease')
    for index, data in enumerate(values):
        if data!=0:
            plt.text(x=index, y=data+0.1, s=f'{data}', fontdict=dict(fontsize=10))

    for index, data in enumerate(all_values):
        plt.text(x=index, y=data+0.1, s=f'{data}', fontdict=dict(fontsize=10))
    plt.legend()
    return f 

def get_mlflow_exp_name(mlflow_exp_name:str, is_binary_classification:bool =False,  multi_label: bool = False):
    """Set the mlflow experiment name based on the task

    Args:
        mlflow_exp_name (str): experiment name from config
        is_binary_classification (bool, optional):  is the task a binary classification. Defaults to False.
        multi_label (bool, optional): is the task a multi-label classification. Defaults to False.

    Returns:
        _type_: _description_
    """ 
    try:
        if mlflow_exp_name is None:
            if is_binary_classification:
                exp_name = 'Binary - Acne Classifier'
            elif multi_label:
                exp_name = 'Multi-Label Acne Classifier'
            else: 
                exp_name = 'Acne Classifier'
        else: 
            exp_name = mlflow_exp_name
    except: 
        exp_name = 'Experiment'

    return exp_name 


def get_top_k_classification_report(df_preds: pd.DataFrame, k: int, classes: list):
    """Generate top-k classification report

    Args:
        df_preds (pd.DataFrame): predictions
        classes (list): list of classes

    Returns:
        dict: top-k classification report
    """        
    exp_name= 'top'+str(k)+'_prediction'
    prob_columns = ['prob_' + disease for disease in classes]
    preds_list = []
    for i in range(k):
        str_name = 'Pred'+ str(i+1)    
        preds_list.append(str_name)
        df_preds[str_name] =df_preds[prob_columns].apply(lambda x: x.sort_values(ascending=False).index[i].replace('prob_', ''), axis=1)

    try:
        df_preds['labels'] = df_preds['label_names'].apply(lambda x: x[0])
    except:
        df_preds['labels'] = df_preds['label']

    df_preds[exp_name] = df_preds.apply(lambda row: bool(set([row['labels']]).intersection(set(row[preds_list].values))), axis=1).reset_index(drop=True)
    try:
        print(f"{exp_name} Accuracy: {df_preds[exp_name].value_counts()[True]/df_preds.shape[0]:0.3f}%.")
    except:
        pass
    y_pred = df_preds.apply(lambda x : x['labels'] if x[exp_name] else x['Pred1'] , axis=1).to_list()
    y_score = df_preds.apply(lambda x : x['labels'], axis=1).to_list()
    report = classification_report( y_score,y_pred, digits=4, output_dict=True)
    return pd.DataFrame(report).transpose()



    