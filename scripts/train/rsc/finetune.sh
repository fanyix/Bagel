#!/bin/bash
MOUNT_ROOT=/checkpoint/maestro
TIMESTAMP=$1

CODE_DIR=$(pwd)
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

export BAGEL_DATA_ROOT="$MOUNT_ROOT/datasets"
export MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)

MODEL_PATH="$MOUNT_ROOT/models/BAGEL-7B-MoT"
OUTPUT_DIR="$MOUNT_ROOT/wam_runs/bagel/$TIMESTAMP"
PROC_PER_NODE=8

# Get the rank of this job
(( ALL_RANKS = PROC_PER_NODE * SLURM_NNODES ))

# Finetune BAGLE-A7B
torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$PROC_PER_NODE \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_NODE \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $MODEL_PATH \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --use_wandb False \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 2 \
  --finetune_from_hf True \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --results_dir "$OUTPUT_DIR" \
  --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
  --num_shard $ALL_RANKS
