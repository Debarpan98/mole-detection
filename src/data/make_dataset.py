
""" Module that created dataset
    Functions:
        ratio_check_resized_gcs -- checks the ratio of resized images from Google Cloud Bucket
        read_images_from_folder -- read images from folder
        read_images_from_db -- read images from database
        read_images_gcs -- read images from google cloud system
        read_images_initial -- read and save test and train sets in csv file

"""
from sklearn.decomposition import dict_learning
import torch
import pandas as pd
from tqdm import tqdm
from src.data.skin_functions import calculate_skin_percentage_general_new
from src.utils.gcs_utils import *
from src.utils.utils import save_csv_general
from src.data.GCPDataset import GCPDataset
from src.utils import envs 
from src.data.dataset_creation_factory import DatasetCreation, DatasetCreationFactory
e=envs.Envs()


def generate_idx_dataset(transform:dict, df:pd.DataFrame, mask_skin: bool, classes:list,train_ids:list, test_ids:list, batch_size:int=16,
                        num_workers:int =4, multi_label:bool=False): 
    """Create datasets and dataloaders from train and test idnexes

    Args:
        transform (dict): dictionary of torchivision transforms
        df (pd.dataFrame): dataset
        train_ids (list): list of training indexes
        test_ids (list): list of test indexes
        batch_size (int, optional):batch size. Defaults to 16.
        img_src_type (int, optional): image source type. Defaults to 2.
        num_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        dataloaders (dict): train and test torch.utils.data.dataloader.DataLoader
        image_datasets (dict): dictionary of train et test datasets (src.data.GCPDataset.GCPDataset')
    """    

    image_datasets = {}

    df_train = df.loc[train_ids, :].reset_index(drop=True)
    df_test = df.loc[test_ids, :].reset_index(drop=True)

    image_datasets["train"] = GCPDataset(df_dataset=df_train, mask_skin= mask_skin, classes=classes,
                                        transform= transform['train'], mutli_label=multi_label)
    image_datasets["test"] = GCPDataset(df_dataset=df_test, mask_skin= mask_skin, classes=classes,
                                        transform= transform['test'], mutli_label=multi_label)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers,  pin_memory=True, persistent_workers=True  ) for x in ['train', "test"]}

    return dataloaders, image_datasets


def create_df_dataset(images:list, labels:list, classes: list):
    images = pd.DataFrame(images, columns=['filename'])
    labels=pd.DataFrame(labels, columns=classes)
    return pd.concat([images, labels], axis=1).reset_index()

def generate_split_data(transform:dict,  df_dataset:pd.DataFrame, mask_skin: bool,  classes:list, random_state:int=42,  batch_size:int=16, 
                        num_workers:int =4, test_percentage:float =0.2, dataset_creator:DatasetCreation=None,
                        test_set_from_file:bool =False,test_set_path:str = None, val_set_from_file:bool =False,
                        val_set_path:str = None,train_set_from_file:bool =False,train_set_path:str = None, 
                        val_set:bool = False, one_tag:bool =False):
    """Generates the dataloaders and datasets

    Args:
        transform (dict): dictionary of torchivision transforms
        df_dataset (pd.DataFrame): dataset
        random_state (int, optional): seed . Defaults to 42.
        batch_size (int, optional):  batch size. Defaults to 16.
        img_src_type (int, optional): image src type. Defaults to 2.
        num_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        dataloaders (dict): train and test torch.utils.data.dataloader.DataLoader
        image_datasets (dict): dictionary of train et test datasets (src.data.GCPDataset.GCPDataset')
    """


    image_datasets = {}
    dataloaders= {}
    # df_val is None when val_set==False
    sets = dataset_creator.generate_split_data(df_dataset, classes,random_state, test_percentage, test_set_from_file,
                                               test_set_path, val_set_from_file, val_set_path,train_set_from_file, train_set_path, val_set=val_set,
                                                one_tag =one_tag)

    for set_key in sets.keys():
        if set_key == 'test':
            image_datasets[set_key] = GCPDataset(df_dataset=sets[set_key], mask_skin= mask_skin, classes=classes,
                                            transform= transform[set_key], mutli_label=dataset_creator.multi_label, gcs_image_bucket='oro-ds-test-bucket')
        else:
            image_datasets[set_key] = GCPDataset(df_dataset=sets[set_key], mask_skin= mask_skin, classes=classes,
                                            transform= transform[set_key], mutli_label=dataset_creator.multi_label)
        
        
        if set_key =='val' or set_key=='test':
            shuffle=False
        else: 
            shuffle=True

        dataloaders[set_key] = torch.utils.data.DataLoader(image_datasets[set_key], batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=True  )

    return dataloaders, image_datasets



def get_images_skin_ratio(config:dict, df_final: pd.DataFrame)->pd.DataFrame:
    """Compute skin ratio for each image

    Args:
        config (dict): dictionary of configuration settings
        df_final (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: updated dataset with threshold ratio
    """    
    num_img_not_resized = 0
    #num_img_under_thresold= 0
    ratios = []

    for index, row in tqdm(df_final.iterrows(),  total=df_final.shape[0]):
        image_read=None
        
        value=blob_exists(filename= row['pathBucketImage'], bucket_img= config.get('gcs_bucket_img'))
        if value:
            bucket = get_bucket_gcs(config.get('gcs_bucket_img'))
            try:
                image_read = cv2.resize(cv2.imdecode(np.asarray(bytearray(bucket.blob(row['pathBucketImage']).download_as_string()), dtype=np.uint8), cv2.IMREAD_COLOR),(config.get('size')[0], config.get('size')[1]))
            except Exception as ex:
                print('Error occured', 'image is:', row['pathBucketImage'])
                df_final.drop(index, inplace=True)
            if image_read is not None:
                ratio = calculate_skin_percentage_general_new(image_read)
                ratios.append(ratio)
            else:
                df_final.drop(index, inplace=True)
        else:
            num_img_not_resized+=1
            df_final.drop(index, inplace=True)

    print(f"Number of images not resized: {num_img_not_resized}")
    df_final['ratios'] = ratios
    return df_final

def read_images_from_db(config: dict_learning)->pd.DataFrame: 
    """Reads images from database
    Args:
        config (dict): dictionary of configuration settings
    """    
    conn = connect2derminator(host = config.get("pg_host_derminator"), 
                              database=config.get("pg_database_derminator"),
                              e=e)
    cur = conn.cursor()
    read = """ select * from derminator_api.label """
    cur.execute(read)
    records = cur.fetchall()
    col_names = []
    for elt in cur.description:
        col_names.append(elt[0])

    cur.close()
    conn.close()
    df = pd.DataFrame(records, columns=col_names)
    df = DatasetCreationFactory().create_dataset(general_dataset=True).create_dataset(df)
    #Remove images starts with /resized to eliminate double count
    df=df[~df.pathBucketImage.str.contains("resized")] 
    print('Number of images on database:', df.shape[0])
    df.loc[:,'pathBucketImage'] = 'resized/' + df['pathBucketImage'].astype(str)
    df = get_images_skin_ratio(config, df)
    path = Path(config.get("gcs_folder")) / config.get('gcs_dataset_folder') / f"dataset_{config.get('dataset_version')}.csv"
    save_csv_general(df, path)
    print(f"Dataset has successfully been saved: {path}")

def merge_acne(df):
    classes_acne =  ['acne_cystic', 'acne_excoriated', 'acne_comedos', 'acne_mixed']
    masks = df.apply(lambda x: x[classes_acne ].any(),  axis=1)
    df['acne'] = masks
    df.drop(columns=classes_acne, inplace=True)
    return df

def get_dataset(config):
    return  pd.read_csv(f'gs://{config.get("gcs_bucket")}/{config.get("gcs_folder")}/{config.get("gcs_dataset_folder")}/dataset_{config.get("dataset_version")}.csv', index_col=0)

def read_images_initial(df_final: pd.DataFrame, classes: list, binary_classification: bool = False,multi_label:bool = False, merge_acne: bool = False ):

    dataset_creator = DatasetCreationFactory().create_dataset(is_binary_classification=binary_classification, is_multi_label=multi_label)
    if merge_acne:
        df_final = merge_acne(df_final)

    df_final = dataset_creator.create_dataset(df_final, classes)  
    df_final =df_final.rename(columns={'pathBucketImage': 'filename'})
    print(f'Dataset contains {df_final.shape[0]} images.')
    return df_final
   
