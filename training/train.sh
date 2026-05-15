#!/bin/bash

source H:/Projects/spring-2026-hmhornung-MinecraftVAE/.venv_local/scripts/activate

JOB_NAME="MC-VAE-NEW-DATASET"

python --version

python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 --nnodes=1 --nproc_per_node=1 H:/Projects/spring-2026-hmhornung-MinecraftVAE/text2mc-source-code/training/train_ddp_torchrun_L40s.py --job_name "$JOB_NAME"
