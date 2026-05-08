import torch
import torch.nn as nn
import numpy as np
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
import os
import json
import h5py
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from MinecraftVAEDataset import MinecraftVAEDataset
from MinecraftVAEDataset import collate_fn

from torch.utils.data import Dataset, DataLoader

def loss_function(embeddings_pred, block_air_pred, x, mu, logvar, data_tokens, air_token_id, epsilon=1e-8):
    # embeddings_pred and x: (Batch_Size, Embedding_Dim, D, H, W)
    # block_air_pred: (Batch_Size, 1, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Ensure embeddings_pred and x have matching spatial dimensions
    assert embeddings_pred.shape == x.shape, f"Shape mismatch: embeddings_pred {embeddings_pred.shape}, x {x.shape}"

    # Move Embedding_Dim to the last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1).contiguous()
    x = x.permute(0, 2, 3, 4, 1).contiguous()

    # Flatten spatial dimensions
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W

    embeddings_pred_flat = embeddings_pred.view(N, embedding_dim)
    x_flat = x.view(N, embedding_dim)

    # Flatten data_tokens
    data_tokens_flat = data_tokens.view(-1)

    # Prepare labels for Cosine Embedding Loss
    y = torch.ones(N, device=x_flat.device)

    # Compute Cosine Embedding Loss per voxel without reduction
    cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    loss_per_voxel = cosine_loss_fn(embeddings_pred_flat, x_flat, y)

    # Mask out the error tensor to include only the errors of the non-air blocks
    mask = (data_tokens_flat != air_token_id)
    loss_per_voxel = loss_per_voxel[mask]

    # Compute mean over non-air blocks
    num_non_air_voxels = loss_per_voxel.numel() + epsilon
    recon_loss = loss_per_voxel.sum() / num_non_air_voxels

    # Prepare ground truth labels for block vs. air
    block_air_labels = (data_tokens != air_token_id).float()

    # Compute binary cross-entropy loss
    bce_loss_fn = nn.BCELoss()
    block_air_pred_probs = block_air_pred.squeeze(1)
    bce_loss = bce_loss_fn(block_air_pred_probs, block_air_labels)

    # Compute KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses
    total_loss = recon_loss + bce_loss + KLD

    return total_loss, recon_loss, bce_loss, KLD

# Function to convert embeddings back to tokens
def embedding_to_tokens(embeddings_pred, embedding_matrix):
    # embeddings_pred: PyTorch tensor of shape (Batch_Size, Embedding_Dim, D, H, W)
    # embedding_matrix: NumPy array of shape (Num_Tokens, Embedding_Dim)

    # Move Embedding_Dim to last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1).contiguous()  # Shape: (Batch_Size, D, H, W, Embedding_Dim)
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W

    # Flatten embeddings
    embeddings_pred_flat = embeddings_pred.view(-1, embedding_dim).cpu().numpy()  # Shape: (N, Embedding_Dim)

    # Normalize embeddings_pred_flat
    embeddings_pred_flat_norm = embeddings_pred_flat / (np.linalg.norm(embeddings_pred_flat, axis=1, keepdims=True) + 1e-8)

    # Normalize embedding_matrix
    embedding_matrix_norm = embedding_matrix / (np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    cosine_similarity = np.dot(embeddings_pred_flat_norm, embedding_matrix_norm.T)  # Shape: (N, Num_Tokens)

    # Find the token with the highest cosine similarity
    tokens_flat = np.argmax(cosine_similarity, axis=1)  # Shape: (N,)

    # Reshape tokens back to (Batch_Size, D, H, W)
    tokens = tokens_flat.reshape(batch_size, D, H, W)
    tokens = torch.from_numpy(tokens).long()  # Convert to torch tensor

    return tokens

# Function to interpolate and generate builds
def interpolate_and_generate(encoder,
                             decoder,
                             dataset,
                             save_dir,
                             epoch,
                             device,
                             num_interpolations=40,
                             dims=(32,32,32)
                             ):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Load the two builds
        # dataset = MinecraftVAEDataset(
        #     data_path="../../data/MinecraftVAEDataset/",
        #     sample_names=[build1_path, build2_path],
        #     dims=dims,
        #     mask_threshold=
        # )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        data_list = []
        data_tokens_list = []
        for data, data_tokens in data_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            data_list.append(data)
            data_tokens_list.append(data_tokens)

        z_list = []
        for data in data_list:
            z, mu, logvar = encoder(data)
            z_list.append(z)

        # Interpolate between z1 and z2
        z1 = z_list[0]
        z2 = z_list[1]

        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolations.append(z_interp)

        # Generate builds from interpolated latent vectors
        for idx, z in enumerate(interpolations):
            embeddings_pred, block_air_pred = decoder(z)
            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(embeddings_pred, dataset.token2vector).to(device)
            # Apply block-air mask
            block_air_pred_labels = (block_air_pred.squeeze(1) >= 0.5).long()
            air_mask = (block_air_pred_labels == 0)
            # Assign air_token_id to air voxels
            recon_tokens[air_mask] = dataset.block2token["minecraft:air"]
            # Convert to numpy array
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (Depth, Height, Width)

            # Save the interpolated build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')

def block_air_metrics(block_air_pred, data_tokens, air_token_id):
    # Prepare block-air predictions and labels
    block_air_pred_probs = block_air_pred.squeeze(1)  # Shape: (Batch_Size, D, H, W)
    block_air_pred_labels = (block_air_pred_probs >= 0.5).long()
    block_air_labels = (data_tokens != air_token_id).long()

    # Flatten tensors
    block_air_pred_flat = block_air_pred_labels.view(-1).cpu()
    block_air_labels_flat = block_air_labels.view(-1).cpu()

    # Compute classification metrics
    accuracy = (block_air_pred_flat == block_air_labels_flat).sum().item() / block_air_labels_flat.numel()
    precision = precision_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
    recall = recall_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
    f1 = f1_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)

    return accuracy, precision, recall, f1

def get_paths_dict(copy_to_node, job_name):
    # Get paths to home directory, regardless of if we're copying the project onto the node's disk
    home_dir = '/fastscratch/hmhornung/spring-2026-hmhornung-minecraftvae/text2mc-source-code/'

    # Model weight checkpoints
    home_train_results_path = os.path.join(home_dir, f'MinecraftVAE_train_results/{job_name}')
    os.makedirs(home_train_results_path, exist_ok=True)
    checkpoint_path = os.path.join(home_train_results_path, 'checkpoint.pth')
    best_model_path = os.path.join(home_train_results_path, 'best_model.pth')

    # Interpolations
    home_interpolations_path = os.path.join(home_train_results_path, 'interpolations/')
    os.makedirs(home_interpolations_path, exist_ok=True)
    save_dir = home_interpolations_path
    os.makedirs(save_dir, exist_ok=True)
    path_dict = {
        "home_dir": home_dir,
        "checkpoint_path": checkpoint_path,
        "best_model_path": best_model_path,
        "home_interpolations_path": home_interpolations_path,
        "save_dir": save_dir
    }

    if copy_to_node:
        # Get project directory on node
        node_proj_path = f'/tmp/spring-2026-hmhornung-minecraftvae/text2mc-source-code'

        # Model weight checkpoints
        node_train_results_path = os.path.join(node_proj_path, f'train_results/{job_name}')
        os.makedirs(node_train_results_path, exist_ok=True)
        
        path_dict["node_checkpoint_path"] = os.path.join(node_train_results_path, 'checkpoint.pth')
        path_dict["node_best_model_path"] = os.path.join(node_train_results_path, 'best_model.pth')

        # Interpolations
        node_interpolations_path = os.path.join(node_train_results_path, 'interpolations/')
        os.makedirs(node_interpolations_path, exist_ok=True)
        path_dict["node_save_dir"] = node_interpolations_path

        # Data
        node_data_path = os.path.join(node_proj_path, 'data/MinecraftVAEDataset/')
        path_dict["data_path"] = node_data_path
        path_dict["block2token_filepath"] = os.path.join(node_data_path, 'block2token.json')
        path_dict["builds_folder_path"] = os.path.join(node_data_path, 'samples/')
        path_dict["build1_path"] = 'batch_319_8281.npy'
        path_dict["build2_path"] = 'batch_225_5840.npy'
        path_dict["token2vector"] = os.path.join(node_data_path, 'embeddings.json')
    else:
        # Data
        home_data_path = os.path.join(home_dir, 'data/MinecraftVAEDataset/')
        path_dict["data_path"] = home_data_path
        path_dict["block2token_filepath"] = os.path.join(home_data_path, 'block2token.json')
        path_dict["builds_folder_path"] = os.path.join(home_data_path, 'samples/')
        path_dict["build1_path"] = 'batch_319_8281.npy'
        path_dict["build2_path"] = 'batch_225_5840.npy'
        path_dict["token2vector"] = os.path.join(home_data_path, 'token2vector.npy')
    
    return path_dict