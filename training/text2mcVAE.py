import torch
from torch import nn
from torch.nn import functional as F
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder

class text2mcVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = text2mcVAEEncoder()
        self.decoder = text2mcVAEDecoder()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        embeddings_pred, block_air_pred = self.decoder(z)
        return z, mu, logvar, embeddings_pred, block_air_pred