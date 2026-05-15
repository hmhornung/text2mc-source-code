# decoder.py

import torch
from torch import nn
from torch.nn import functional as F

# Reuse AxialAttentionBlock3D and text2mcVAEResidualBlock from encoder.py
from encoder import AxialAttentionBlock3D, text2mcVAEResidualBlock

class text2mcVAEDecoder(nn.Module):
    def __init__(self, embedding_dim:int=32, scale:int=3, latent_size:int=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        init_channels = 2 ** scale
        self.initial_layers = nn.Sequential(
            nn.ConvTranspose3d(latent_size, init_channels*128, kernel_size=4, stride=4, padding=0),  # D=8 -> D=32
            text2mcVAEResidualBlock(init_channels*128, init_channels*128),
            AxialAttentionBlock3D(init_channels*128),
            text2mcVAEResidualBlock(init_channels*128, init_channels*128),
        )
        self.upsampling_layers = nn.Sequential(
            nn.ConvTranspose3d(init_channels*128, init_channels*64, kernel_size=2, stride=2, padding=0),  # D=32 -> D=64
            text2mcVAEResidualBlock(init_channels*64, init_channels*64),
            text2mcVAEResidualBlock(init_channels*64, init_channels*64),
        )
        self.shared_layers = nn.Sequential(
            nn.GroupNorm(32, init_channels*64),
            text2mcVAEResidualBlock(init_channels*64, init_channels*64),
        )
        # First head: embeddings
        self.embedding_head = nn.Sequential(
            nn.Conv3d(init_channels*64, embedding_dim, kernel_size=3, padding=1),
        )
        # Second head: block vs. air prediction
        self.block_air_head = nn.Sequential(
            nn.Conv3d(init_channels*64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upsampling_layers(x)
        x = self.shared_layers(x)
        embeddings = self.embedding_head(x)
        block_air_prob = self.block_air_head(x)
        return embeddings, block_air_prob
