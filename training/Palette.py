import amulet
import os
import sys
import json
import numpy as np
import pandas as pd
from amulet import load_level
from amulet_nbt import load
from amulet.api.selection import SelectionBox
from collections import namedtuple
import re
import mcschematic
from gdpc import Block
from gdpc.block import transformedBlockOrPalette
import itertools

class Palette:
    
    def __init__(self, src: str | dict | list):
        # Get the block to token mapping
        if isinstance(src, str):
            with open(src, 'r') as file:
                self.block2token = json.load(file)
        elif isinstance(src, dict):
            self.block2token = src
        elif isinstance(src, list):
            self.block2token = {k : i for i, k in enumerate(src)}
        else:
            raise ValueError("src argument should be a file path (str), block2token mapping (dict), or list of blocks (list)")

        # Validate that all block strings are valid
        self._blockstr_pattern = re.compile(r"^minecraft:[a-z0-9_]+(?:\[[a-z0-9_]+=[a-z0-9_]+(?:,[a-z0-9_]+=[a-z0-9_]+)*\])?$")
        self._blockstr_separation_pattern = pattern = re.compile(r"(minecraft:[^\[]+)(?:\[(.*)\])?")
        
        invalid_blocks = []
        for blockstr in self.block2token.keys():
            if not self._valid_blockstr(blockstr):
                invalid_blocks.append(blockstr)
        
        if invalid_blocks:
            raise ValueError(f"The following block strings are invalid: {invalid_blocks}")
            
        # Get other mapping lists
        self.block_strings = [blockstr for blockstr in self.block2token.keys()]
        self.token2block = {v:k for k,v in self.block2token.items()}
        self.gdpc_blocks = [self._blockstr_to_gdpc_block(blockstr) for blockstr in self.block_strings]
        self.length = len(self.block_strings)
        
        self._add_missing_transformations()
        
        self.TransformLookup = namedtuple('TransformLookup', ['rot0', 'rot0flip', 'rot90', 'rot90flip', 'rot180', 'rot180flip', 'rot270', 'rot270flip'])
        lookups = self._generate_transformation_lookups()
        self.transform_lookup = self.TransformLookup(*lookups)
        
        
    def reduce_blockstates(self, keep_blockstates: list, block_ids=None) -> tuple["Palette", np.ndarray, np.ndarray]:
        reduced_block2tok = {}
        reduced_gdpc_blocks = []
        src2tgt_lookup = np.zeros(self.length, dtype=np.int16)
        
        if block_ids is None: block_ids = []
        
        for blockstr, token in self.block2token.items():
            block_id, states = self._blockstr_to_id_states(blockstr)
            # Case 1: no block ids specified, strip blockstates from any block
            # Case 2, block ids specified, strip blockstates from only those specified
            if not block_ids or block_id in block_ids:
                # remove any blockstate info were not keeping
                if states: states = {k:v for k,v in states.items() if k in keep_blockstates}

            # Using GDPC block objects since they can tell if two blocks with different state orders are equal
            reduced_block = Block(block_id, states)
            
            # add to the new palette if its not already in
            if reduced_block not in reduced_gdpc_blocks:
                reduced_gdpc_blocks.append(reduced_block)
                reduced_block2tok[str(reduced_block)] = len(reduced_block2tok)
            
            # Now, map this block to the reduced palette
            reduced_token = reduced_block2tok[str(reduced_block)]
            src2tgt_lookup[token] = reduced_token
        
        # Now, we reverse the lookup table so we can convert back to the original palette. The tgt tokens will map to the first instance in the src that maps to it
        tgt2src_lookup = np.full(len(reduced_block2tok), -1, dtype=np.int16)
        
        for i, token in enumerate(src2tgt_lookup):
            if tgt2src_lookup[token] == -1:
                tgt2src_lookup[token] = i
        
        # Now, construct the new Palette
        reduced_palette = Palette(reduced_block2tok)
        
        return reduced_palette, src2tgt_lookup, tgt2src_lookup
    
    def get_mask_lookup(self, mask_list = np.ndarray | list, mask_block:str = "minecraft:air"):
        air_token = self.block2token[mask_block]
        
        if isinstance(mask_list, np.ndarray):
            mask_arr = mask_list.astype(bool)
        
        elif isinstance(mask_list, list):
            # Using base names for masking
            mask_arr = np.zeros((self.length), dtype=bool)
            for block_id in mask_list:
                block_strings = [str(block) for block in self.gdpc_blocks if block.id == block_id]
                for block_str in block_strings:
                    mask_arr[self.block2token[block_str]] = True
        
        # Map all masked tokens to the air token
        src2mask = np.arange(self.length, dtype=np.int32)
        src2mask[mask_arr] = air_token
        
        return src2mask
    
    def _generate_transformation_lookups(self):
        rotations = [0,1,2,3]
        flips = [0,1]
        combos = list(itertools.product(rotations, flips))
        lookup_arrays = []
        
        for rotation, flip in combos:
            transformed_blocks = transformedBlockOrPalette(block=self.gdpc_blocks, rotation=rotation, flip=(flip,0,0))
            lookup_arr = np.full(self.length, -1, dtype=np.int16)
            for i, block in enumerate(transformed_blocks):
                lookup_arr[i] = self.block2token[str(block)]
            
            assert -1 not in lookup_arr
            lookup_arrays.append(lookup_arr)
        
        return lookup_arrays
            
    def _add_missing_transformations(self):
        count_added = 0
        rotations = [0,1,2,3]
        flips = [0,1]
        combos = list(itertools.product(rotations, flips))
        
        for rotation, flip in combos:
            all_possible_blocks = transformedBlockOrPalette(block=self.gdpc_blocks, rotation=rotation, flip=(flip,0,0))
            for block in all_possible_blocks:
                if block not in self.gdpc_blocks:
                    blockstr = str(block)
                    self.gdpc_blocks.append(block)
                    self.block_strings.append(blockstr)
                    self.block2token[blockstr] = len(self.block2token)
                    self.token2block[self.block2token[blockstr]] = blockstr
                    count_added += 1
        self.length = len(self.block_strings)
        print(f'Added {count_added} blocks for potential tranformations')
                
    
    def _valid_blockstr(self, blockstr: str) -> bool:
        return bool(self._blockstr_pattern.fullmatch(blockstr))
    
    def _blockstr_to_id_states(self, blockstr: str):
        block_id, states_str = self._blockstr_separation_pattern.fullmatch(blockstr).groups()
        states = dict(p.split("=") for p in states_str.split(",")) if states_str else None
        return block_id, states
    
    def _blockstr_to_gdpc_block(self, blockstr):
        block_id, states = self._blockstr_to_id_states(blockstr)
        return Block(block_id, states)
    
    def __str__(self):
        lines = [f"{token}: {block}" for block, token in self.block2token.items()]
        return '\n'.join(lines)
    
    def __len__(self):
        return self.length
    