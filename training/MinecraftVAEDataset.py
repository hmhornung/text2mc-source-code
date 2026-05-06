import os
import sys
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

sys.path.insert(0, '../../data_processing/palette/')
from palette import Palette


class MinecraftVAEDataset(Dataset):
    def __init__(self, data_path:str):
        self.block_dist_df = pd.read_csv(os.path.join(data_path, 'block_dist_dataframe.csv'))
        self.sample_dims_df = pd.read_csv(os.path.join(data_path, 'sample_dims_dataframe.csv'))

        sample_files = os.listdir(os.path.join(data_path, 'samples/'))
        sample_names = self.sample_dims_df["Sample"].list()
        
        
        with open(os.path.join(data_path, 'block2token.json')) as f:
            self.block2token = json.load(f)
        
        self.palette = Palette(self.block2token)
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pass

    def _transform(self, sample_arr, flip:bool, rotate:int):
        pass
