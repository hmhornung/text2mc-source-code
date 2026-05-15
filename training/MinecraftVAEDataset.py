import os
import sys
import json
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from Palette import Palette
from util import array_to_schematic

class MinecraftVAEDataset(Dataset):
    def __init__(self, data_path:str, sample_names:list, dims:tuple=(32,32,32), mask_threshold:int=0, precision:np.dtype=np.float64):
        # Load necessary files and data
        self.data_path = data_path
        self.sample_names = sample_names
        self.mask_threshold = mask_threshold
        self.dims = np.array(dims, dtype=np.int64)
        self.precision = precision
        
        self.sample_dims_df = pd.read_csv(os.path.join(data_path, 'sample_dims_df.csv'))
        self.exclude_samples = self.sample_dims_df[self.sample_dims_df["volume"] < 100]["Sample"].tolist()
        self.exclude_samples = [sample + '.npy' for sample in self.exclude_samples]
        self.sample_names = [sample for sample in self.sample_names if sample not in self.exclude_samples]
        
        self.sample_dir = os.path.join(data_path, 'samples/')
        # self.sample_names = self.sample_dims_df["Sample"].tolist()
        self.token2vector = np.load(os.path.join(data_path, "token2vector.npy")).astype(self.precision)
        self.emb2game = np.load(os.path.join(data_path, "emb2game_lookup.npy"))
        
        with open(os.path.join(data_path, 'block2token.json')) as f:
            self.block2token = json.load(f)
            
        with open(os.path.join(data_path, 'game_block2token.json')) as f:
            self.game_block2token = json.load(f)
        self.game_token2block = {v:k for k,v in self.game_block2token.items()}
        
        # Create the Palette from the block2token, as well as blockstate-free Palette
        self.palette = Palette(self.block2token)
        self.palette_no_blockstate, self.bs_src2tgt, self.bs_tgt2src = self.palette.reduce_blockstates(keep_blockstates=[])
        
        # Load samples
        self.block_counts = np.zeros(len(self.palette_no_blockstate), dtype=np.int64)
        
        self.samples = []
        
        for sample in self.sample_names:
            arr = np.load(os.path.join(self.sample_dir, sample)).astype(np.int16)
            self.samples.append(arr)
            
            # Get the block counts
            no_bs_arr = self.bs_src2tgt[arr]
            blocks, counts = np.unique(no_bs_arr, return_counts=True)
            self.block_counts[blocks] += counts
        
        self.rand_aug = True
        self.weighted_sampling = True
        self._heatmaps_created = False
        
        self.mask_percentile(self.mask_threshold)
    
    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.weighted_sampling:
            center = self._weighted_random_coordinate(self.heatmaps[idx])
            sample = self._get_window(sample, center)
        else:
            center = [random.randrange(0,n) for n in sample.shape]
            sample = self._get_window(sample, center)
            
        if self.rand_aug:
            rotate = random.randint(0, 3)
            flip = random.choice([True, False])
            sample = self._transform(sample, flip, rotate)
            
        return {"emb": self.token2vector[sample], "tok": sample}

    def _transform(self, arr, flip:bool, rotate:int):
        if flip:
            arr = np.flip(arr, axis=0)
            match rotate:
                case 0:
                    arr = self.palette.transform_lookup.rot0flip[arr]
                case 1:
                    arr = self.palette.transform_lookup.rot90flip[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
                case 2:
                    arr = self.palette.transform_lookup.rot180flip[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
                case 3:
                    arr = self.palette.transform_lookup.rot270flip[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
        else:
            match rotate:
                case 0:
                    arr = self.palette.transform_lookup.rot0[arr]
                case 1:
                    arr = self.palette.transform_lookup.rot90[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
                case 2:
                    arr = self.palette.transform_lookup.rot180[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
                case 3:
                    arr = self.palette.transform_lookup.rot270[arr]
                    arr = np.rot90(arr, k=rotate, axes=(0,2))
        return arr
    
    def _weighted_random_coordinate(self, heatmap):
        flat = heatmap.ravel().astype(np.float64)
        p = flat / flat.sum()
        idx = np.random.choice(flat.size, p=p)
        coord = np.unravel_index(idx, heatmap.shape)
        return coord
    
    def _get_window(self, array, coordinate):
        pad = np.stack([self.dims//2, self.dims//2], axis=1)
        padded = np.pad(array, pad_width=pad, mode="constant", constant_values=self.palette.block2token["minecraft:air"])
        slice_x, slice_y, slice_z = (slice(d,d+p) for d, p in zip(coordinate, self.dims))
        return padded[slice_x, slice_y, slice_z]
    
    def mask_percentile(self, percentile:int):
        # Get the blocks to mask below the percentile
        threshold = np.percentile(self.block_counts, percentile)
        bool_mask = self.block_counts < threshold
        self.mask_blocks = []
        for i, mask in enumerate(bool_mask):
            if mask:
                self.mask_blocks.append(self.palette_no_blockstate.token2block[i])
        
        self.block_mask = self.palette.get_mask_lookup(self.mask_blocks)
        
        self.block_counts = np.zeros(len(self.palette_no_blockstate), dtype=np.int64)
        
        # Mask the blocks in the dataset
        for i, sample in enumerate(self.samples):
            self.samples[i] = self.block_mask[sample]
            
            # Get the block counts
            no_bs_arr = self.bs_src2tgt[self.samples[i]]
            blocks, counts = np.unique(no_bs_arr, return_counts=True)
            self.block_counts[blocks] += counts
        
        # Re-calculate the block densities and heatmaps:
        self.block_densities = np.log( 1 + (self.block_counts.sum() / (self.block_counts + 1)) ).astype(np.float16)
        self.block_densities[self.palette_no_blockstate.block2token["minecraft:air"]] = 1e-3
        self.heatmaps = []
        for sample in self.samples:
            self.heatmaps.append(self.block_densities[self.bs_src2tgt[sample]].astype(np.float16))
        
        self._heatmaps_created = True
    
    def plot_block_occurences(self):
        block_counts_dist = np.sort(self.block_counts)

        fig, ax = plt.subplots()

        # Styling
        ax.set_title("Block Occurences (Log Scale)")
        ax.set_ylabel("# Occurences")
        ax.set_xlabel("Block Token IDs (Sorted by frequency)")


        ax.plot(block_counts_dist)
        ax.set_yscale('log')
    
    def arr2schem(self, array, save_path, filename):
        array_to_schematic(self.emb2game[array], self.game_token2block, save_path, filename)
        
    
def collate_fn(batch):
    embed = torch.tensor(np.stack([x["emb"] for x in batch])).permute(0,4,1,2,3)
    tokens = torch.tensor(np.stack([x["tok"] for x in batch]))
    return embed, tokens
    