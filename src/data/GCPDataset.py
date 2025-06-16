import numpy as np
import torch
import time
import cv2

from PIL import Image
from torch.utils.data import Dataset
from src.utils.gcs_utils import get_bucket_gcs
from sklearn import preprocessing
from src.data.skin_functions import masknonskin
import pandas as pd
from src.utils import envs 
e=envs.Envs()

class GCPDataset(Dataset):

    def __init__(self, df_dataset:pd.DataFrame, mask_skin: bool, classes:list, mutli_label:bool=False, transform=None, gcs_image_bucket: str = e.gcs_image_bucket):
        self.targets=classes
        self.targets.sort()
        self.df_dataset=df_dataset.reset_index(drop=True)
        self.transform = transform
        self.le = preprocessing.LabelEncoder()
        self.le.fit(df_dataset[self.targets].columns)
        self.multi_label = mutli_label
        self.mask_skin = mask_skin

        self.gcs_image_bucket = gcs_image_bucket

    def __len__(self):
        return self.df_dataset.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df_dataset.loc[idx,:]
        img_name = row['filename']
        if self.multi_label:
            labels = torch.FloatTensor(row[self.targets].tolist())
        else:
            labels = row[self.targets].tolist().index(1)
#        image = self.read_image(img_name, self.mask_skin)
        image = self.read_image(img_name, self.mask_skin, self.gcs_image_bucket) 
        if self.transform:
            image = self.transform(image)
        return image, labels, img_name


    def read_image(self, img_name, mask_skin, gcs_image_bucket):
        # read images
#        bucket = get_bucket_gcs(e.gcs_image_bucket)

        bucket = get_bucket_gcs(gcs_image_bucket)

        success = False
        counter = 0
        while (success == False):
            try:
                image=cv2.imdecode(np.asarray(bytearray(bucket.blob(img_name).download_as_string()), dtype=np.uint8),cv2.IMREAD_COLOR)
                print(image.shape)
                success = True
            except:
                if (counter > 10):
                    success = True
                time.sleep(2)
                counter += 1
                print("Loading Error occured from bucket on image", img_name, "number of tries:", counter)

        try:
            if mask_skin:
                image = masknonskin(image, "either", 1)
            #print(image
        except Exception as ex:
            print('Error occured in resnext resize and mask', 'image is:', img_name)
            print(ex)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print(image)
        except Exception as ex:
            print('Error occured in resnext bgr2rgb', 'image is:', img_name)
            print(ex)

        return Image.fromarray(image)