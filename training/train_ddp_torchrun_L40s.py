import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import os
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
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAE import text2mcVAE

from text2mc_train_utils import loss_function
from text2mc_train_utils import embedding_to_tokens
from text2mc_train_utils import interpolate_and_generate
from text2mc_train_utils import block_air_metrics
from text2mc_train_utils import get_paths_dict

parser = argparse.ArgumentParser()
parser.add_argument("--job_name", type=str)
args = parser.parse_args()

job_name = args.job_name
batch_size = 2
num_epochs = 1
fixed_size = (64, 64, 64)
mask_threshold = 0
embedding_dim = 32
copy_to_node = True
num_workers = 4
prefetch_factor = 3
metric_workers = 6
lr = 1e-5
epochs = 48
# batch_limit = 250

paths = get_paths_dict(copy_to_node, job_name)

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl", world_size=world_size, rank=local_rank, device_id=local_rank)

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
        home_checkpoint_path: str
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.gpu_id}"
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.air_token_id = air_token_id
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.checkpoint_path = checkpoint_path
        self.home_checkpoint_path = home_checkpoint_path
        self.rank = dist.get_rank()
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint")
            self._load_checkpoint(checkpoint_path)
        dist.barrier()

        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)
        
        if dist.get_rank() == 0:
            self.executor = ThreadPoolExecutor(max_workers=metric_workers)
            self.pending_futures = []
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

        print(f"Trainer initialized on GPU {self.gpu_id}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)

            if self.rank == 0:
                if self.total_overall_loss < self.best_loss:
                    self.best_loss = self.total_overall_loss
                    # self._save_checkpoint(epoch, self.checkpoint_path)
                    self._save_checkpoint(epoch+1, self.home_checkpoint_path)
                
                self._save_checkpoint(epoch+1, f"{self.home_checkpoint_path.removesuffix('.pth')}_epoch_{epoch+1}.pth")

    def _run_epoch(self, epoch):
        b_sz = batch_size
        self.train_data.sampler.set_epoch(epoch)
        if self.rank == 0:
            print(f"\n-----TRAINING-----\n")
            print(f"TRAIN Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        # TRAINING
        self.model.train()
        num_batches = 0
        if self.rank == 0:
            self._reset_metric_totals()
        
        start_time = time.perf_counter()

        for batch_idx, (data, data_tokens) in enumerate(self.train_data):
            data        = data.to(self.device)
            data_tokens = data_tokens.to(self.device)
            overall_loss, reconstruction_loss, bce_loss, KL_divergence, block_air_pred_all, data_tokens_all = self._run_batch(data, data_tokens, train=True)
            
            if dist.get_rank() == 0:
                batch_time = time.perf_counter()
                if num_batches % 100 == 0: print(f"TRAIN Epoch {epoch} | Batch {num_batches} Compute done at {(batch_time - start_time):.3f} sec")
                # SUBMIT the async metrics to the executor
                
                future = self.executor.submit(
                    self._end_of_batch_metrics,
                    overall_loss,
                    reconstruction_loss,
                    bce_loss, KL_divergence,
                    block_air_pred_all,
                    data_tokens_all,
                    num_batches,
                    epoch
                )
                self.pending_futures.append(future)

                # CHECK for complete metrics
                still_pending = []

                for f in self.pending_futures:
                    if f.done():
                        # If done, get the results, print the batche's metrics, and add to the running total
                        ovr_loss_result, recon_loss_result, bce_loss_result, KL_result, acc_result, prc_result, rcl_result, f1_result, batch_num_result, epoch_num_result = f.result()
                        metric_time = time.perf_counter()
                        if batch_num_result % 100 == 0: print(f"TRAIN Epoch {epoch_num_result} | Batch {batch_num_result} Metric done at {(metric_time - start_time):.3f} sec")
                        print(f"TRAIN Epoch {epoch_num_result} | Batch {batch_num_result} | TOTAL {ovr_loss_result:.4f} | RECON {recon_loss_result:.4f} | BCE {bce_loss_result:.4f} | KL {KL_result:.4f} | ACC {acc_result:.4f} | PRC {prc_result:.4f} | RCL {rcl_result:.4f} | F1 {f1_result:.4f}")

                        self.total_overall_loss += ovr_loss_result
                        self.total_reconstruction_loss += recon_loss_result
                        self.total_bce_loss += bce_loss_result
                        self.total_KL_divergence += KL_result
                        self.total_accuracy += acc_result
                        self.total_precision += prc_result
                        self.total_recall += rcl_result
                        self.total_f1 += f1_result
                    else:
                        still_pending.append(f)
                    
                self.pending_futures = still_pending

            num_batches += 1

        # End of Training batches, wait for metrics to be asynchronously computed
        if dist.get_rank() == 0:

            wait(self.pending_futures)

            for f in self.pending_futures:
                ovr_loss_result, recon_loss_result, bce_loss_result, KL_result, acc_result, prc_result, rcl_result, f1_result, batch_num_result, epoch_num_result = f.result()
                print(f"TRAIN Epoch {epoch_num_result} | Batch {batch_num_result} | TOTAL {ovr_loss_result:.4f} | RECON {recon_loss_result:.4f} | BCE {bce_loss_result:.4f} | KL {KL_result:.4f} | ACC {acc_result:.4f} | PRC {prc_result:.4f} | RCL {rcl_result:.4f} | F1 {f1_result:.4f}")

                self.total_overall_loss += ovr_loss_result
                self.total_reconstruction_loss += recon_loss_result
                self.total_bce_loss += bce_loss_result
                self.total_KL_divergence += KL_result
                self.total_accuracy += acc_result
                self.total_precision += prc_result
                self.total_recall += rcl_result
                self.total_f1 += f1_result
            
            self.pending_futures.clear()

            self.total_overall_loss /= num_batches
            self.total_reconstruction_loss /= num_batches
            self.total_bce_loss /= num_batches
            self.total_KL_divergence /= num_batches
            self.total_accuracy /= num_batches
            self.total_precision /= num_batches
            self.total_recall /= num_batches
            self.total_f1 /= num_batches

            print(f"\nTRAIN AVG Epoch {epoch} | TOTAL {self.total_overall_loss:.4f} | RECON {self.total_reconstruction_loss:.4f} | BCE {self.total_bce_loss:.4f} | KL {self.total_KL_divergence:.4f} | ACC {self.total_accuracy:.4f} | PRC {self.total_precision:.4f} | RCL {self.total_recall:.4f} | F1 {self.total_f1:.4f}\n")
        
        dist.barrier()

        # VALIDATION
        b_sz = len(next(iter(self.val_data))[0])
        self.val_data.sampler.set_epoch(epoch)
        if self.rank == 0: 
            print(f"\n-----VALIDATION-----\n")
            print(f"VAL Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}\n")

        self.model.eval()

        num_batches = 0
        if self.rank == 0:
            self._reset_metric_totals()

        for batch_idx, (data, data_tokens) in enumerate(self.val_data):
            data        = data.to(self.device)
            data_tokens = data_tokens.to(self.device)
            with torch.no_grad():
                overall_loss, reconstruction_loss, bce_loss, KL_divergence, block_air_pred_all, data_tokens_all = self._run_batch(data, data_tokens, train=False)

            if dist.get_rank() == 0:
                # SUBMIT the async metrics to the executor
                
                future = self.executor.submit(
                    self._end_of_batch_metrics,
                    overall_loss,
                    reconstruction_loss,
                    bce_loss, KL_divergence,
                    block_air_pred_all,
                    data_tokens_all,
                    num_batches,
                    epoch
                )
                self.pending_futures.append(future)

                # CHECK for complete metrics

                still_pending = []

                for f in self.pending_futures:
                    if f.done():
                        # If done, get the results, print the batche's metrics, and add to the running total
                        ovr_loss_result, recon_loss_result, bce_loss_result, KL_result, acc_result, prc_result, rcl_result, f1_result, batch_num_result, epoch_num_result = f.result()
                        print(f"VAL Epoch {epoch_num_result} | Batch {batch_num_result} | TOTAL {ovr_loss_result:.4f} | RECON {recon_loss_result:.4f} | BCE {bce_loss_result:.4f} | KL {KL_result:.4f} | ACC {acc_result:.4f} | PRC {prc_result:.4f} | RCL {rcl_result:.4f} | F1 {f1_result:.4f}")

                        self.total_overall_loss += ovr_loss_result
                        self.total_reconstruction_loss += recon_loss_result
                        self.total_bce_loss += bce_loss_result
                        self.total_KL_divergence += KL_result
                        self.total_accuracy += acc_result
                        self.total_precision += prc_result
                        self.total_recall += rcl_result
                        self.total_f1 += f1_result
                    else:
                        still_pending.append(f)
                    
                self.pending_futures = still_pending

            num_batches += 1
        
        # End of Validation batches, wait for metrics to be asynchronously computed
        if dist.get_rank() == 0:

            wait(self.pending_futures)

            for f in self.pending_futures:
                ovr_loss_result, recon_loss_result, bce_loss_result, KL_result, acc_result, prc_result, rcl_result, f1_result, batch_num_result, epoch_num_result = f.result()
                print(f"VAL Epoch {epoch_num_result} | Batch {batch_num_result} | TOTAL {ovr_loss_result:.4f} | RECON {recon_loss_result:.4f} | BCE {bce_loss_result:.4f} | KL {KL_result:.4f} | ACC {acc_result:.4f} | PRC {prc_result:.4f} | RCL {rcl_result:.4f} | F1 {f1_result:.4f}")

                self.total_overall_loss += ovr_loss_result
                self.total_reconstruction_loss += recon_loss_result
                self.total_bce_loss += bce_loss_result
                self.total_KL_divergence += KL_result
                self.total_accuracy += acc_result
                self.total_precision += prc_result
                self.total_recall += rcl_result
                self.total_f1 += f1_result
            
            self.pending_futures.clear()

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
            embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=self.air_token_id
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

        # Create a target list for gathering all process tensors into
        if dist.get_rank() == 0:
            block_air_pred_gather = [torch.zeros_like(block_air_pred, device=self.device) for i in range(dist.get_world_size())]
            data_tokens_gather = [torch.zeros_like(data_tokens, device=self.device) for i in range(dist.get_world_size())]
        else:
            block_air_pred_gather = None
            data_tokens_gather = None
        
        # concat them into one tensor
        dist.gather(tensor=block_air_pred, gather_list=block_air_pred_gather, dst=0)
        dist.gather(tensor=data_tokens, gather_list=data_tokens_gather, dst=0)
        # Sum the losses, much simpler
        dist.reduce(overall_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(reconstruction_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(bce_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(KL_divergence, dst=0, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            block_air_pred_all = torch.cat(block_air_pred_gather, dim=0).cpu() # Take off of gpu before starting async
            data_tokens_all = torch.cat(data_tokens_gather, dim=0).cpu()
        else:
            block_air_pred_all = None
            data_tokens_all = None
        
        return overall_loss.item(), reconstruction_loss.item(), bce_loss.item(), KL_divergence.item(), block_air_pred_all, data_tokens_all

    def _reset_metric_totals(self):
        self.total_overall_loss = 0
        self.total_reconstruction_loss = 0
        self.total_bce_loss = 0
        self.total_KL_divergence = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0

    def _end_of_batch_metrics(self, overall_loss, reconstruction_loss, bce_loss, KL_divergence, block_air_pred_all, data_tokens_all, batch_num, epoch_num):
        
        with torch.no_grad():
            accuracy, precision, recall, f1 = block_air_metrics(block_air_pred_all, data_tokens_all, self.air_token_id)
        
        return overall_loss, reconstruction_loss, bce_loss, KL_divergence, accuracy, precision, recall, f1, batch_num, epoch_num
        
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint["EPOCHS_RUN"]
        self.best_loss = checkpoint["VAL_LOSS"]
        print(f"\nResuming training from checkpoint at Epoch {self.epochs_run}\nCheckpoint Path: {checkpoint_path}\nValidation Loss: {self.best_loss}\n")
    
    def _save_checkpoint(self, epoch, path):
        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "VAL_LOSS": self.best_loss,
        }
        torch.save(checkpoint, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}\n")

def load_train_objs():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    sample_files = os.listdir(paths["builds_folder_path"])
    test_files = [paths["build1_path"], paths["build2_path"]]
    sample_files = [sample for sample in sample_files if sample not in test_files]
    
    # Split the file paths into training, validation, and test sets
    dataset_size = len(sample_files)
    
    validation_split = 0.2
    train_size = int((1 - validation_split) * dataset_size)
    val_size = int(validation_split * dataset_size)
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
    
    model = text2mcVAE(embedding_dim=embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return train_dataset, val_dataset, test_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=shuffle, seed=42, drop_last=True),
        collate_fn=collate_fn
    )

def main(save_every: int, total_epochs: int, batch_size: int, checkpoint_path: str, home_checkpoint_path: str):
    ddp_setup()
    train_dataset, val_dataset, test_dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size, shuffle=True)
    val_data = prepare_dataloader(val_dataset, batch_size)
    trainer = Trainer(model, train_data, val_data, train_dataset.air_token, optimizer, save_every, checkpoint_path, home_checkpoint_path)
    trainer.train(total_epochs)
    destroy_process_group()
    print('did trainer.destroy_process_group')

if __name__ == "__main__":
    main(1, epochs, batch_size, paths["node_checkpoint_path"], paths["checkpoint_path"])