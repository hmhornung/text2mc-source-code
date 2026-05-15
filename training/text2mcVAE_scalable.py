import torch
from torch import nn
from torch.nn import functional as F
from encoder_scalable import text2mcVAEEncoder
from decoder_scalable import text2mcVAEDecoder

class text2mcVAE(nn.Module):
    def __init__(self, embedding_dim:int=32, scale:int=3, latent_size:int=8):
        super().__init__()
        self.encoder = text2mcVAEEncoder(embedding_dim, scale, latent_size)
        self.decoder = text2mcVAEDecoder(embedding_dim, scale, latent_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        embeddings_pred, block_air_pred = self.decoder(z)
        return z, mu, logvar, embeddings_pred, block_air_pred