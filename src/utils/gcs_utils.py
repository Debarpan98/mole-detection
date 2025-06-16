""" Helper functions for Google Cloud System
    Helpers with buckets and databases.

Functions:
   connect2derminator -- create a new database session
   get_bucket_gcs -- get the google bucket object
   blob_exists -- verify is blob exists
"""
import psycopg2
import time
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import torch

from pathlib import Path
from google.cloud import storage

def set_environement_variable():
    if "GCS" not in os.environ:
        print("Using Default Env Variable:GCS")
        os.environ["GCS"]="true"
    if "LOAD_MODELS" not in os.environ:
        print("Using Default Env Variable:LOAD_MODELS")
        os.environ["LOAD_MODELS"] = "false"
    if "GCS_IMAGE_BUCKET" not in os.environ:
        print("Using Default Env Variable:GCS_IMAGE_BUCKET")
        os.environ["GCS_IMAGE_BUCKET"] = "oro-ds-test-bucket"
    if "USER_DERMINATOR" not in os.environ:
        print("Using Default Env Variable:USER_DERMINATOR")
        os.environ["USER_DERMINATOR"] = "dKcMDvfRzRYVhmYN2na3"
    if "PW_DERMINATOR" not in os.environ:
        print("Using Default Env Variable:PW_DERMINATOR")
        os.environ["PW_DERMINATOR"] = "fyRtl0c3SLgWxvJIXFNSAsFh6n2nwi0RRb"
    # mlflow env
    if "MLFLOW_TRACKING_URI" not in os.environ:
        print("Using Default Env Variable:MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow-gcp-2xsb65gbpa-nn.a.run.app"
    if "MLFLOW_TRACKING_USERNAME" not in os.environ:
        print("Using Default Env Variable:MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
    if "MLFLOW_EXPERIMENT_NAME" not in os.environ:
        print("Using Default Env Variable:MLFLOW_EXPERIMENT_NAME")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "mole_public_dataset"
    if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
        print("Using Default Env Variable:MLFLOW_TRACKING_PASSWORD")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = 'Ubs&".eiILe)rIL:'
    #    raise Exception("no MLFLOW_TRACKING_PASSWORD has been passed as env varibale")
    return True

def connect2derminator(host, database, e):
    """Create a new database session

    Args:
        host (str): pg host dermintaor
        database (str): pg database dermintaor
        e: envs

    Returns: 
        new connection object
    """
    return psycopg2.connect(
    host=host,
    database=database,
    user=e.user_derminator,
    port=5433,
    password=e.pw_derminator)


def get_bucket_gcs(bucket_name):
    """Get the Google Cloud bucket

    Args:
        bucket_name (str): bucket name

    Returns:
        bucket object
    """
    counter = 0
    success=False
    while (success == False):
        try:
            bucket = storage.Client().get_bucket(bucket_name)
            success = True
        except Exception as e:
            #print(e)
            if (counter > 10):
                success = True
            time.sleep(2)
            counter += 1
            print("Read error problem on connecting to bucket", "number of tries:", counter)
    return bucket


def blob_exists(filename, bucket_img):
    """Verify if blob already exists

    Args:
        filename (str): blob filename
        bucket_img (str) : gcs bucket image

    Returns:
        [boolean]: boolean indicating if blob exists
    """
    bucket = get_bucket_gcs(bucket_img)
    blob = bucket.blob(filename)
    return blob.exists()

def image_reader(filename, bucket_img):
    """Read an image from a bucket

    Args:
        filename (str): blob filename
        bucket_img (str) : gcs bucket image


    Returns:
        [array]: image
    """
    bucket = get_bucket_gcs(bucket_img)
    image_read = cv2.cvtColor(cv2.resize(cv2.imdecode(np.asarray(bytearray(bucket.blob(filename).download_as_string()), dtype=np.uint8),cv2.IMREAD_COLOR),(224,224)),cv2.COLOR_BGR2RGB)
    return image_read


def save_yml_file(bucket_name, model_path_gcs, file):
    df = pd.DataFrame([file.keys(), file.values()]).T
    df_csv_data= df.to_csv(encoding="utf-8")
    bucket = get_bucket_gcs(bucket_name)
    bucket.blob(model_path_gcs).upload_from_string(df_csv_data,content_type='text/csv')
    
def save_from_filename(bucket_name, model_path_gcs, model_output_dir):
    """Save file from filename path into google bucket

    Args:
        bucket_name (str): bucket name
        model_path_gcs (str): destination path
        model_output_dir (str): path to file
    """    
    bucket = get_bucket_gcs(bucket_name)
    blob = bucket.blob(model_path_gcs)
    blob._chunk_size = 8388608  # 1024 * 1024 B * 16 = 8 MB
    blob.upload_from_filename(model_output_dir)
    print(f'Saving to {model_path_gcs} from {model_path_gcs}')

def load_from_filename(bucket_name, model_path_gcs, model_output_dir ):
    bucket = get_bucket_gcs(bucket_name)
    folder = Path(model_output_dir).parent
    folder.mkdir(parents=True, exist_ok= True)
    if blob_exists(model_path_gcs, bucket):
        print(f'Loading {model_path_gcs} from gcs.')
        blob = bucket.blob(model_path_gcs)
        blob._chunk_size = 8388608  # 1024 * 1024 B * 16 = 8 MB
        blob.download_to_filename(model_output_dir)
    else:
        print('Store.db does not exist.')
def read_images_gcs(bucket, prefix='shanel_test/data_4class_skin_diseases/'):
    #set_gcs_env(gcs=1) #TODO
    images=[]
    labels=[]

    iterator = bucket.list_blobs(delimiter="/", prefix=prefix)
    response = iterator._get_next_page_response()
    
    for c in response['prefixes']:
        if not c.endswith("/"):
            c += "/"

        iterator_inside = bucket.list_blobs(delimiter="/", prefix=c)
        response_inside = iterator_inside._get_next_page_response()

        num_of_images=len(response_inside['items'])
        #print(num_of_images)

        i=0
        while i<num_of_images:
            img_bucket_uri=response_inside['items'][i]['name']

            images.append(img_bucket_uri)
            labels.append(c.split('/')[-2])
            i +=1

    #print(len(images),len(labels))
    # set_gcs_env(gcs=0) #TODO
    return images,labels

def image_reader(bucket, image):
    image_read = cv2.cvtColor(cv2.resize(cv2.imdecode(np.asarray(bytearray(bucket.blob(image).download_as_string()), dtype=np.uint8),cv2.IMREAD_COLOR),(224,224)),cv2.COLOR_BGR2RGB)
    return image_read


def save_model(model, bucket_name:str, model_path_gcs:Path, mlflow_id: str):
    """Saves model state dictionnary to google bucket

    Args:
        model (nn.Module): model
        bucket_name (str): bucket name 
        model_path_gcs (str): path to bucket 
    """
    model_name = model.__class__.__name__
    name = f'{model_name}-{mlflow_id}.pth'
    torch.save(model.state_dict(), name)
    save_from_filename(bucket_name, str(model_path_gcs/ name), name)
    os.remove(name)