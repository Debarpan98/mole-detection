from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from src.utils.utils import sample_with_max_value
from sklearn.model_selection import StratifiedKFold
from src.utils.utils import sample_with_max_value, remove_images_skin_ratio, increase_with_max_value
import numpy as np
import pandas as pd

class DatasetCreation(ABC):
    """
    Factory for creating the dataset
    """
    def __init__(self, check_ratio:bool, subsamble_set:bool, skin_ratio_threshold:float, subsample_max_values:int, subsample_max_per_disease:dict ):
        """
        Args:
            check_ratio (bool): indicating if we discard the images with skin ratio < skin_ratio_threshold
            subsamble_set (bool): indicating if we subsample the dataset
            skin_ratio_threshold (float): skin ratio threshold
            subsample_max_values (int): default maximum of images per disease
            subsample_max_per_disease (dict): dictionary of diseases with maximum of images (if disease not in dict, then use default value)
        """  
        # other classes are all the relevant classes used in the current 41 diseases classification
        self.other_classes = ['rosacea_inflammatory', 'atopic_dermatitis', 'rosacea_erythemato_telangiectasique','peri_oral_dermatitis',
                'seborrheic_keratosis','psoriasis_vulgar','seborrheic_dermatitis','nummular_eczema',
                'tinea_versicolor','chronic_hand_eczema','vulgar_warts','folliculitis','alopecia_androgenic',
                'dyshidrosis','nevus','melasma','alopecia_areata','intertrigo','urticaria','vitiligo','keratosis_pilaris',
                'molluscum','cheilitis_eczematous','tinea_corporis','prurigo_nodularis','actinic_keratosis',
                'genital_warts','plane_warts','pityriasis_rosae','melanonychia','psoriasis_pustular_palmoplantar',
                'granuloma_annulare','psoriasis_guttate','lichen_simplex_chronicus','shingles','herpes_simplex']

        # all acne classes
        self.classes_acne =  ['acne_cystic', 'acne_scars', 'acne_excoriated', 'acne_comedos', 'acne_mixed',]
        self.check_ratio = check_ratio
        self.subsamble_set= subsamble_set
        self.skin_ratio_threshold = skin_ratio_threshold
        self.subsample_max_values = subsample_max_values
        self.subsample_max_per_disease = subsample_max_per_disease 

    
    def get_all_columns(self):
        """ Returns all the disease columns

        Returns:
            list: list of all diseases
        """        
        all_columns = self.other_classes.copy()
        all_columns.extend(self.classes_acne.copy())
        return all_columns
    
    def get_test_train_set_from_file(self, df_dataset: pd.DataFrame, test_set_path:str):
        """Generates train and test set from test set files.
           All the images from 'test_set_path' are goin in the test set.
           The rest of the images are going in the training set. 
      
        Args:
            df_dataset (pd.DatFrame): dataset
            test_set_path (str): google cloud path to file. For example: gs://oro-ds-test-bucket/sdd_acne_files/models/dl/results/test_set_initial.csv'

        Returns:
            df_test, df_train: test dataframe and train dataframe
        """        
        print(f'Using existing test file: {test_set_path}')
        df_test =  pd.read_csv(test_set_path)
        masks =df_dataset['filename'].isin(df_test['filename'].unique())
        df_test = df_dataset[masks]       
        df_train = df_dataset[~masks]   
        return df_train, df_test

    def check_skin_and_subsample_dataset(self, df: pd.DataFrame):
        """Check skin ratio and remove images with skin ratio < threshold
           Then, subsample dataset.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: updated dataset
        """        
        if self.check_ratio:
            print(f'Removing images with skin ratio < {self.skin_ratio_threshold}.')
            df = remove_images_skin_ratio(self.skin_ratio_threshold, df)    
        if self.subsamble_set:
            print('Subsampling dataset')
            df = sample_with_max_value(df, self.subsample_max_values, self.subsample_max_per_disease)
        return df

    def describe_dataset(self, df:pd.DataFrame, classes:list, mode:str='train'):
        """Print description of dataset

        Args:
            df (pd.DataFrame): dataset
            classes (list): list of classes
            mode (str, optional): mode ['test', 'train', 'val']. Defaults to 'train'.
        """        
        print(f'There are {df.shape[0]} images in {mode} test.')
        print(f'Final {mode} set:\n{df[classes].sum()}')

    
    def check_one_tag_images(self, df: pd.DataFrame, classes:list, one_tag:bool)->pd.DataFrame:
        """_Remove images with more than one tag (more than one label) when one_tage is True
        Args:
            df (pd.DataFrame): dataset
            classes (list): list of classes
            one_tag (bool): boolean indicating if we want to keep images with only one tag 

        Returns:
            pd.DataFrame: updated dataset 
        """        
        init_shape = df.shape[0]
        if one_tag:
            df = df.loc[(df[classes].sum(axis=1) == 1), :].reset_index(drop=True)
            print(f'We removed {init_shape-df.shape[0]} images with more than 1 tag.')
        return df

    def create_set_from_file(self,df_dataset: pd.DataFrame, set_path:str):
        """Create a dataset from a csv file

        Args:
            df_dataset (pd.DataFrame): all dataset (all images available)
            set_path (str): path to csv file (used to create dataset)

        Returns:
            dataset : dataset created from set_path file
        """        
        print(f'Using existing file: {set_path}')
        df =  pd.read_csv(set_path)
        masks=df_dataset['filename'].isin(df['filename'].unique())
        return df_dataset[masks] 

    def create_all_sets_from_files(self, df_dataset: pd.DataFrame, train_set_path: str, test_set_path: str, val_set_path: str, classes: list, val_set: bool ):
        """Create datasets from files

        Args:
            df_dataset (pd.DataFrame): dataset of all images in database
            train_set_path (str): path to training set csv file
            test_set_path (str): path to test set csv file
            val_set_path (str): path to val set csv file
            classes (list): list of classes
            val_set (bool): boolean indicating if we need a validation set

        Returns:
            dict: dictionnary of train, test and val sets
        """        
        sets ={}
        df_train = self.create_set_from_file(df_dataset, train_set_path)
        df_test = self.create_set_from_file(df_dataset, test_set_path)
        used_images = pd.concat([df_train['filename'], df_test['filename']]).unique()
        if val_set:
            df_val = self.create_set_from_file(df_dataset, val_set_path)
            used_images= np.append(used_images, df_val['filename'])
            self.describe_dataset( df_val, classes, mode='val')
            sets['val'] = df_val
        masks =df_dataset['filename'].isin(used_images)
        df_not_used = df_dataset[~masks]       
        if self.check_ratio:
            df_not_used = df_not_used[df_not_used['ratios'] >= self.skin_ratio_threshold].reset_index(drop=True)
        df_train = increase_with_max_value(df_not_used.reset_index(drop=True) , df_train.reset_index(drop=True),  
                                          self.subsample_max_values, self.subsample_max_per_disease)   
        sets['test']= df_test
        sets['train'] = df_train
        self.describe_dataset( df_test, classes, mode='test')
        self.describe_dataset( df_train, classes, mode='train')
        return sets


    def generate_split_data(self, df_dataset: pd.DataFrame, classes: list, random_state: int, test_percentage: float, 
                            test_set_from_file:bool, test_set_path: str, val_set_from_file:bool, val_set_path:str,
                            train_set_from_file:bool,train_set_path:str, val_set: bool, one_tag: bool):
        """Split in test and train set

        Args:
            df_dataset (pd.DataFrame): dataset
            classes (list): classes
            random_state (int): seed
            test_percentage (float): test percentage
            test_set_from_file (str): boolean indicating if we want to create test set from file
            test_set_path (str): path to test set csv file
            val_set_from_file (str): boolean indicating if we want to create val set from file
            val_set_path (str): path to val set csv file
            train_set_from_file (str): boolean indicating if we want to create train set for file 
            train_set_path (str): path to training set csv file
            val_set (bool): boolean indicating if we need a validation set
            one_tag (bool): boolean indicating if we only want to keep images with one tag

        Returns:
            dictionary: dictionary of sets
        """ 
        if train_set_from_file:
            sets = self.create_all_sets_from_files(df_dataset, train_set_path,test_set_path, val_set_path, classes, val_set )
        else:
            sets = {}
            if test_set_from_file:
                df_train, df_test =self.get_test_train_set_from_file(df_dataset, test_set_path)
                if val_set:
                    # we need to have a validation set 
                    if val_set_from_file:
                        df_train, df_val =self.get_test_train_set_from_file(df_train, val_set_path)
                        df_train = self.check_one_tag_images(df_train, classes, one_tag)
                        df_train = self.check_skin_and_subsample_dataset(df_train)
                    else:
                        df_train = self.check_one_tag_images(df_train, classes, one_tag)
                        df_train = self.check_skin_and_subsample_dataset(df_train)
                        df_train, df_val = self.iterative_split_set(df_train, classes, test_percentage= test_percentage)
                    
                    self.describe_dataset( df_val, classes, mode='val')
                    sets['val'] = df_val
            else:
                df_dataset = self.check_skin_and_subsample_dataset( df_dataset)
                df_train, df_test= self.iterative_split_set(df_dataset, classes, test_percentage)
                if val_set:
                    # we need to have a validation set 
                    df_train, df_val = self.iterative_split_set(df_train, classes, test_percentage= test_percentage)
                    self.describe_dataset( df_val, classes, mode='val')
                    sets['val'] = df_val
     
            sets['test']= df_test
            sets['train'] = df_train
            self.describe_dataset( df_test, classes, mode='test')
            self.describe_dataset( df_train, classes, mode='train')
        return sets
    
    def create_df_dataset(self, images:list, labels:list, classes: list):
        images = pd.DataFrame(images, columns=['ratios','filename' ])
        labels=pd.DataFrame(labels, columns=classes)
        return pd.concat([images, labels], axis=1).reset_index()

    def iterative_split_set(self, df_dataset: pd.DataFrame, classes: list, test_percentage: float = 0.2):
        """Split dataframe

        Args:
            df_dataset (pd.DataFrame): dataset
            classes (list): list of classes
            test_percentage (float) : test percentage

        Returns:
            (pd.DataFrame,pd.DataFrame): df_train, df_test
        """        
        train_images, train_labels, test_images, test_labels  = iterative_train_test_split(df_dataset[['ratios', 'filename']].to_numpy(),
                                                                                             (df_dataset[classes]*1).to_numpy(), test_size=test_percentage)
        df_train =self.create_df_dataset(train_images, train_labels, classes)
        df_test =self.create_df_dataset(test_images, test_labels, classes)
        return df_train, df_test
        
    

    @abstractmethod
    def create_dataset(self, df_dataset: pd.DataFrame, classes: list):
        pass

    @abstractmethod
    def create_folds(self, df_dataset, classes, k_folds, random_state=None):
        pass


class GeneralDataset(DatasetCreation):


    def create_dataset(self, df: pd.DataFrame):
        """Create dataset for ALL tasks
        The dataset contains all the images tagged with one or more of the 40 classes

        Args:
            df (pd.DataFrame): dataset read from database

        Returns:
            _pd.DataFrame: dataset with only images tagged with one or more of the 40 classes
        """    
        all_columns = self.get_all_columns()
        return df.loc[(df[all_columns].sum(axis=1) >0),:].reset_index(drop=True)

    def generate_split_data(self):
        pass
    def create_folds(self):
        pass

class MultiLabelAcneDataset(DatasetCreation):
    def __init__(self,  check_ratio:bool, subsamble_set:bool, skin_ratio_threshold:float, subsample_max_values:int, subsample_max_per_disease:dict):
        DatasetCreation.__init__(self, check_ratio, subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease)
        self.multi_label = True

    def create_dataset(self, df:pd.DataFrame, classes:list):
        """Create dataset

        Args:
            df (pd.DataFrame): dataset
            classes (list): classification classes
            all_classes (list): the classes
        Returns:
            (pd.DataFrame): updated dataset
        """

        all_classes = df.columns
        all_classes = all_classes.drop(['pathBucketImage', 'ratios', 'labelledby', 'dateInserted', 'reviewedby'])
        mask = df.apply(lambda x: x[classes].any(),  axis=1)
        df= df[mask].reset_index()
        other_columns = set(all_classes) - set(classes)
        df = df.loc[(df[other_columns].sum(axis=1) == 0),:].reset_index(drop=True)
        columns = set(classes)
        columns.add('pathBucketImage') 
        columns.add('ratios') 
        df_final = df[columns].reset_index(drop=True)
        df_final = df_final.loc[(df_final[classes].sum(axis=1) > 0),:]
        return df_final
    
    def create_folds(self,  df_dataset: pd.DataFrame, classes: list, k_folds:int, random_state:int = None):
        """Create folds

        Args:
            df_dataset (pd.DataFrame): dataset
            classes (list): classes
            k_folds (int): number of folds
            random_state (int, optional): seed. Defaults to None.

        Returns:
            (folds, np.array): folds, array of labels
        """        
        np_labels = (df_dataset[classes]*1).to_numpy()
        kfold = IterativeStratification(n_splits=k_folds, order=1)
        return kfold, np_labels


class SingleLabelAcneDataset(DatasetCreation):
    def __init__(self,  check_ratio:bool, subsamble_set:bool, skin_ratio_threshold:float, subsample_max_values:int, subsample_max_per_disease:dict):
        DatasetCreation.__init__(self, check_ratio, subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease)
        self.multi_label = False


    def create_dataset(self, df:pd.DataFrame, classes:list):
        """Create dataset

        Args:
            df (pd.DataFrame): dataset
            classes (list): classification classes
            all_classes (list): the classes
        Returns:
            (pd.DataFrame): updated dataset
        """     
        df = df.loc[(df[classes].sum(axis=1) == 1), :]
        columns = set(classes)
        columns.add('pathBucketImage') 
        columns.add('ratios')
        return df[columns].reset_index(drop=True)

    def create_folds(self,  df_dataset, classes, k_folds, random_state):
        np_labels = (df_dataset[classes]*1).to_numpy()
        kfold = StratifiedKFold(n_splits=k_folds, shuffle = True, random_state=random_state)
        np_labels = np_labels.argmax(axis=1)
        return kfold, np_labels
    

# class BinaryAcneDataset(DatasetCreation):
#     def __init__(self,  check_ratio:bool, subsamble_set:bool, skin_ratio_threshold:float, subsample_max_values:int, subsample_max_per_disease:dict):
#         DatasetCreation.__init__(self, check_ratio, subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease)
#         self.multi_label = False
    
    
#     def generate_split_data(self, df_dataset: pd.DataFrame, classes: list, random_state: int, test_percentage: float, 
#                             test_set_from_file:bool, test_set_path: str, val_set_from_file:bool, val_set_path:str,
#                             val_set:bool=False, one_tag:bool =False):
#         """Split in test and train set

#         Args:
#             df_dataset (pd.DataFrame): dataset
#             classes (list): classes
#             random_state (int): seed
#             test_percentage (float): test percentage

#         Returns:
#             (pd.DataFrame, pd.DataFrame): train and test sets
#         """    
#         self.other_classes.append('acne')  
#         sets= {}
#         if test_set_from_file:
#             df_train, df_test =self.get_test_train_set_from_file(df_dataset, test_set_path)
#             df_train = self.check_skin_and_subsample_dataset(df_train)
#             train_images = df_train[['filename', 'diseases']]
#             test_images = df_test[['filename', 'diseases']]
#             train_labels= df_train[self.other_classes]*1
#             test_labels = df_test[self.other_classes]*1
#             train_labels = self.get_acne_labels_from_df( train_labels)
#             test_labels = self.get_acne_labels_from_df(test_labels)
#             df_train =self.create_df_dataset(train_images, train_labels, classes)
#             df_train = self.check_one_tag_images(df_train, classes, one_tag)
#             df_test =self.create_df_dataset(test_images, test_labels, classes)
                        

#         else:
#             df_dataset = self.check_one_tag_images(df_dataset, classes, one_tag)
#             df_dataset = self.check_skin_and_subsample_dataset( df_dataset)
#             df_train, df_test = self.iterative_split_set(df_dataset, classes, test_percentage=  test_percentage)

#         if val_set:
#             # we need to have a validation set 
#             df_train, df_val =  self.iterative_split_set(df_train, classes, test_percentage=  test_percentage)
#             self.describe_dataset( df_val, classes, mode='val')
#             sets['val'] = df_val
     
#         sets['test']= df_test
#         sets['train'] = df_train
#         self.describe_dataset( df_test, classes, mode='test')
#         self.describe_dataset( df_train, classes, mode='train')
#         return sets
        

#     def create_dataset(self, df, classes):
#         """Creates dataset for binary classification.
#            The classification task consists of classifying each
#            image as 'acne' or 'not_acne' .

#         Args:
#             df (pd.DataFrame): dataset
#             classes (list): classification classes
#             all_classes (list): the classes
#         Returns:
#             (pd.DataFrame): updated dataset
#         """ 
#         # binary classification (acne vs not acne )
#         df_binary = df.copy()
#         all_columns =  self.get_all_columns()
        
#         # the diseases column contain 'string' values ( a description of all diseases present in the image)
#         df_binary['diseases'] = df_binary[ all_columns].apply(lambda x: tuple(df_binary[ all_columns].columns[np.argwhere(x.values== True).flatten()], ), axis=1)
#         all_columns.extend(['pathBucketImage', 'diseases', 'ratios'])
#         df_binary = df_binary[all_columns]
#         masks = (df_binary[self.other_classes].sum(axis=1) == 0) & (df_binary[self.classes_acne].sum(axis=1) >0)
#         df_binary['acne'] = masks

#         # keep the images with a least one positive class
#         df_binary = df_binary.loc[(df_binary[self.get_all_columns()].sum(axis=1) >0),:]
#         return df_binary.reset_index(drop=True)

    
#     def create_folds(self,  df_dataset, classes, k_folds, random_state):
#         df_dataset['not_acne'] =  (~df_dataset['acne'].astype(bool)).astype(int)        
#         np_labels = (df_dataset[classes]*1).to_numpy()
#         kfold = IterativeStratification(n_splits=k_folds, order=1)
#         return kfold, np_labels

#     def get_acne_labels_from_df(self, labels:pd.DataFrame):
#         """Get acne labels

#         Args:
#             labels (pd.DataFrame): dataframe of all albels

#         Returns:
#            np.array: array of all labels
#         """        
#         acne_labels = labels['acne']
#         not_acne_labels = (~ acne_labels.astype(bool)).astype(int)                                                                           
#         return np.concatenate([np.expand_dims(acne_labels, axis=1), np.expand_dims(not_acne_labels, axis=1)], axis=1)
    
#     def get_acne_labels(self, labels:np.array, acne_id:int):
#         """Get acne labels

#         Args:
#             labels (np.array): labels of allcolumns
#             acne_id (int): the column number of acne

#         Returns:
#             np.array: array of all labels
#         """        
#         acne_labels = labels[:, acne_id]
#         not_acne_labels = (~ acne_labels.astype(bool)).astype(int)                                                                           
#         return np.concatenate([np.expand_dims(acne_labels, axis=1), np.expand_dims(not_acne_labels, axis=1)], axis=1)

#     def create_df_dataset(self, images:list, labels:list, classes: list):
#         images = pd.DataFrame(images, columns=['filename', 'diseases'])
#         labels=pd.DataFrame(labels, columns=classes)
#         return pd.concat([images.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)

    
#     def iterative_split_set(self, df_dataset: pd.DataFrame, classes: list, test_percentage: float = 0.2):
#         """Split dataframe

#         Args:
#             df_dataset (pd.DataFrame): dataset
#             classes (list): list of classes
#             test_percentage (float) : test percentage

#         Returns:
#             (pd.DataFrame,pd.DataFrame): df_train, df_test
#         """        
#         train_images, train_labels, test_images, test_labels  = iterative_train_test_split(df_dataset[['filename', 'diseases']].to_numpy(), 
#                                                                                          (df_dataset[self.other_classes]*1).to_numpy(), 
#                                                                                          test_size=test_percentage)
#         acne_id = self.other_classes.index('acne') 
#         train_labels = self.get_acne_labels( train_labels ,acne_id)
#         test_labels = self.get_acne_labels( test_labels, acne_id)
      
#         df_train =self.create_df_dataset(train_images, train_labels, classes)
#         df_test =self.create_df_dataset(test_images, test_labels, classes)
#         return df_train, df_test
        

    
class DatasetCreationFactory:
    def create_dataset(self, is_multi_label: bool=False, is_binary_classification:bool=False, general_dataset:bool=False,
                        check_ratio:bool=False, subsamble_set:bool =False, skin_ratio_threshold:float =0.5, subsample_max_values:int=80, 
                        subsample_max_per_disease:dict=None):
        if general_dataset:
            return GeneralDataset()
        elif is_binary_classification:
            raise NotImplemented(f"Acne Binary model no longer supported.")
            #return BinaryAcneDataset(check_ratio,subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease)
        elif is_multi_label:
            return MultiLabelAcneDataset( check_ratio,subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease )
        else:
            return SingleLabelAcneDataset(check_ratio,subsamble_set, skin_ratio_threshold, subsample_max_values, subsample_max_per_disease)







