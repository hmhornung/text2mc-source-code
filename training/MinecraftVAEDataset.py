import os
import sys
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

sys.path.insert(0, '../../data_processing/palette/')
from palette import Palette


class CustomImageDataset(Dataset):
    def __init__(self, data_path:str):
        sample_
        self.block_dist_df = pd.read_csv(os.path.join(data_path, 'block_dist_dataframe.csv'))
        self.sample_dims_df = pd.read_csv(os.path.join(data_path, 'sample_dims_dataframe.csv'))
        
        with open(os.path.join(data_path, 'block2token.json')) as f:
            self.block2token = json.load(f)
        
        self.palette = Palette

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label