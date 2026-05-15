import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn

import numpy as np
import glob
import random
import argparse
from datetime import datetime
import time
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

from MinecraftVAEDataset import MinecraftVAEDataset
from MinecraftVAEDataset import collate_fn
from encoder_scalable import text2mcVAEEncoder
from decoder_scalable import text2mcVAEDecoder
from text2mcVAE_scalable import text2mcVAE

from text2mc_train_utils import loss_function
from text2mc_train_utils import embedding_to_tokens
from text2mc_train_utils import interpolate_and_generate
from text2mc_train_utils import block_air_metrics
from text2mc_train_utils import get_paths_dict

import functools
print = functools.partial(print, flush=True)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        air_token_id: int,
        optimizer: optim.Optimizer,
        save_every: int,
        checkpoint_path: str,
        home_checkpoint_path: str,
        epochs,
    ) -> None:
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.air_token_id = air_token_id
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.epochs=epochs
        self.checkpoint_path = checkpoint_path
        self.home_checkpoint_path = home_checkpoint_path
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint")
            self._load_checkpoint(checkpoint_path)

        # Set up running loss totals for asynchronous accumulation
        self.total_overall_loss = 0
        self.total_reconstruction_loss = 0
        self.total_bce_loss = 0
        self.total_KL_divergence = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0 # <- Great movie
        self.total_f1 = 0
        self.best_loss = float('inf')

        print(f"Trainer initialized on GPU")

    def train(self):
        for epoch in range(self.epochs_run, self.epochs):
            self._run_epoch(epoch)

            if self.total_overall_loss < self.best_loss:
                self.best_loss = self.total_overall_loss
                # self._save_checkpoint(epoch, self.checkpoint_path)
                self._save_checkpoint(epoch+1, self.home_checkpoint_path)
            
            self._save_checkpoint(epoch+1, f"{self.home_checkpoint_path.removesuffix('.pth')}_epoch_{epoch+1}.pth")

    def _run_epoch(self, epoch):
        b_sz = batch_size
        print(f"\n-----TRAINING-----\n")
        print(f"TRAIN Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        # TRAINING
        self.model.train()
        num_batches = 0
        self._reset_metric_totals()
        
        for batch_idx, (data, data_tokens) in enumerate(self.train_data):
            data        = data.to(self.device)
            data_tokens = data_tokens.to(self.device)
            overall_loss, reconstruction_loss, bce_loss, KL_divergence, block_air_pred = self._run_batch(data, data_tokens, train=True)
            
            accuracy, precision, recall, f1 = self._end_of_batch_metrics(block_air_pred, data_tokens)
            
            print(f"TRAIN Epoch {epoch} | Batch {batch_idx} | TOTAL {overall_loss:.4f} | RECON {reconstruction_loss:.4f} | BCE {bce_loss:.4f} | KL {KL_divergence:.4f} | ACC {accuracy:.4f} | PRC {precision:.4f} | RCL {recall:.4f} | F1 {f1:.4f}")

            self.total_overall_loss += overall_loss
            self.total_reconstruction_loss += reconstruction_loss
            self.total_bce_loss += bce_loss
            self.total_KL_divergence += KL_divergence
            self.total_accuracy += accuracy
            self.total_precision += precision
            self.total_recall += recall
            self.total_f1 += f1

            num_batches += 1

        # End of Training batches
        self.total_overall_loss /= num_batches
        self.total_reconstruction_loss /= num_batches
        self.total_bce_loss /= num_batches
        self.total_KL_divergence /= num_batches
        self.total_accuracy /= num_batches
        self.total_precision /= num_batches
        self.total_recall /= num_batches
        self.total_f1 /= num_batches

        print(f"\nTRAIN AVG Epoch {epoch} | TOTAL {self.total_overall_loss:.4f} | RECON {self.total_reconstruction_loss:.4f} | BCE {self.total_bce_loss:.4f} | KL {self.total_KL_divergence:.4f} | ACC {self.total_accuracy:.4f} | PRC {self.total_precision:.4f} | RCL {self.total_recall:.4f} | F1 {self.total_f1:.4f}\n")

        # VALIDATION
        b_sz = len(next(iter(self.val_data))[0])
        
        print(f"\n-----VALIDATION-----\n")
        print(f"VAL Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}\n")

        self.model.eval()

        num_batches = 0
        self._reset_metric_totals()

        for batch_idx, (data, data_tokens) in enumerate(self.val_data):
            data        = data.to(self.device)
            data_tokens = data_tokens.to(self.device)
            with torch.no_grad():
                overall_loss, reconstruction_loss, bce_loss, KL_divergence, block_air_pred = self._run_batch(data, data_tokens, train=False)

            accuracy, precision, recall, f1 = self._end_of_batch_metrics(block_air_pred, data_tokens)
            
            print(f"VAL Epoch {epoch} | Batch {batch_idx} | TOTAL {overall_loss:.4f} | RECON {reconstruction_loss:.4f} | BCE {bce_loss:.4f} | KL {KL_divergence:.4f} | ACC {accuracy:.4f} | PRC {precision:.4f} | RCL {recall:.4f} | F1 {f1:.4f}")

            self.total_overall_loss += overall_loss
            self.total_reconstruction_loss += reconstruction_loss
            self.total_bce_loss += bce_loss
            self.total_KL_divergence += KL_divergence
            self.total_accuracy += accuracy
            self.total_precision += precision
            self.total_recall += recall
            self.total_f1 += f1

            num_batches += 1
        
        # End of Validation batches
        self.total_overall_loss /= num_batches
        self.total_reconstruction_loss /= num_batches
        self.total_bce_loss /= num_batches
        self.total_KL_divergence /= num_batches
        self.total_accuracy /= num_batches
        self.total_precision /= num_batches
        self.total_recall /= num_batches
        self.total_f1 /= num_batches

        print(f"\nVAL AVG Epoch {epoch} | TOTAL {self.total_overall_loss:.4f} | RECON {self.total_reconstruction_loss:.4f} | BCE {self.total_bce_loss:.4f} | KL {self.total_KL_divergence:.4f} | ACC {self.total_accuracy:.4f} | PRC {self.total_precision:.4f} | RCL {self.total_recall:.4f} | F1 {self.total_f1:.4f}\n")
        
    def _run_batch(self, data, data_tokens, train: bool = True):
        if train: self.optimizer.zero_grad()
        # Forward pass
        z, mu, logvar, embeddings_pred, block_air_pred = self.model(data)
        
        # Compute losses
        total_loss, recon_loss, bce_loss, KLD = loss_function(
            embeddings_pred, block_air_pred, data, mu, logvar, data_tokens.contiguous(), air_token_id=self.air_token_id
        )
        
        # Save Losses to return
        overall_loss        = torch.tensor(total_loss.item(), device=self.device)
        reconstruction_loss = torch.tensor(recon_loss.item(), device=self.device)
        bce_loss            = torch.tensor(bce_loss.item(), device=self.device)
        KL_divergence       = torch.tensor(KLD.item(), device=self.device)

        # Backward pass and optimization
        if train:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return overall_loss.item(), reconstruction_loss.item(), bce_loss.item(), KL_divergence.item(), block_air_pred

    def _reset_metric_totals(self):
        self.total_overall_loss = 0
        self.total_reconstruction_loss = 0
        self.total_bce_loss = 0
        self.total_KL_divergence = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0

    def _end_of_batch_metrics(self, block_air_pred_all, data_tokens_all):
        
        with torch.no_grad():
            accuracy, precision, recall, f1 = block_air_metrics(block_air_pred_all, data_tokens_all.contiguous(), self.air_token_id)
        
        return accuracy, precision, recall, f1
        
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint["EPOCHS_RUN"]
        self.best_loss = checkpoint["VAL_LOSS"]
        print(f"\nResuming training from checkpoint at Epoch {self.epochs_run}\nCheckpoint Path: {checkpoint_path}\nValidation Loss: {self.best_loss}\n")
    
    def _save_checkpoint(self, epoch, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "VAL_LOSS": self.best_loss,
        }
        torch.save(checkpoint, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}\n")

def load_train_objs(fixed_size, mask_threshold, lr):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    sample_files = os.listdir(paths["builds_folder_path"])
    test_files = [paths["build1_path"], paths["build2_path"]]
    sample_files = [sample for sample in sample_files if sample not in test_files]
    
    # Split the file paths into training, validation, and test sets
    dataset_size = len(sample_files)
    
    dataset_size_modifier = 1.0
    # dataset_size_modifier = 0.7
    # dataset_size_modifier = 0.365
    # dataset_size_modifier = 0.005
    
    validation_split = 0.1
    train_size = int((1 - validation_split) * (dataset_size * dataset_size_modifier))
    val_size = int(validation_split * (dataset_size* dataset_size_modifier))
    test_size = len(test_files)
    
    # Shuffle file paths
    random.shuffle(sample_files)
    
    # Split the file paths
    train_files = sample_files[:train_size]
    val_files = sample_files[train_size:train_size + val_size]
    
    # Create datasets
    train_dataset = MinecraftVAEDataset(
        data_path=paths["data_path"],
        sample_names=train_files,
        dims=fixed_size,
        mask_threshold=mask_threshold,
        precision=np.float32
    )
    # train_dataset.rand_aug = False
    # train_dataset.weighted_sampling = False
    
    val_dataset = MinecraftVAEDataset(
        data_path=paths["data_path"],
        sample_names=val_files,
        dims=fixed_size,
        mask_threshold=mask_threshold,
        precision=np.float32
    )
    
    test_dataset = MinecraftVAEDataset(
        data_path=paths["data_path"],
        sample_names=test_files,
        dims=fixed_size,
        mask_threshold=mask_threshold,
        precision=np.float32
    )
    
    model = text2mcVAE(embedding_dim=32,
                       scale=1,
                       latent_size=8
                       )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return train_dataset, val_dataset, test_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers, prefetch_factor=1, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        # prefetch_factor=prefetch_factor,
        # persistent_workers=False,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def main(save_every: int, batch_size: int, checkpoint_path: str, home_checkpoint_path: str, fixed_size, mask_threshold, num_workers, prefetch_factor, lr, epochs):
    train_dataset, val_dataset, test_dataset, model, optimizer = load_train_objs(fixed_size, mask_threshold, lr)
    train_data = prepare_dataloader(train_dataset, batch_size, num_workers, prefetch_factor, shuffle=True)
    val_data = prepare_dataloader(val_dataset, batch_size, num_workers, prefetch_factor, shuffle=False)
    trainer = Trainer(model,
                      train_data,
                      val_data,
                      train_dataset.block2token["minecraft:air"],
                      optimizer,
                      save_every,
                      checkpoint_path,
                      home_checkpoint_path,
                      epochs)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str)
    args = parser.parse_args()

    job_name = args.job_name
    batch_size = 18
    fixed_size = (32, 32, 32)
    mask_threshold = 0
    embedding_dim = 32
    copy_to_node = False
    local=True
    num_workers = 0
    prefetch_factor = None
    # lr = 5e-6
    lr = 1e-6
    
    # lr = 5e-9
    # Orig = 1e-5
    epochs = 14
    # batch_limit = 250

    log_file = open(f"{job_name}_output", "w")
    err_file = open(f"{job_name}_error", "w")
    sys.stdout = log_file
    sys.stderr = err_file

    paths = get_paths_dict(copy_to_node, job_name, local=local)
    main(1, batch_size, paths["checkpoint_path"], paths["checkpoint_path"], fixed_size, mask_threshold, num_workers, prefetch_factor, lr, epochs)