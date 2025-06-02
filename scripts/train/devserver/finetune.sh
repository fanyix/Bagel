#!/bin/bash
MOUNT_ROOT=/home/fanyix/data/wam_data
TIMESTAMP=$(date +"%y%m%d%H%M")

CODE_DIR=$(pwd)
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

export BAGEL_DATA_ROOT="$MOUNT_ROOT/datasets"

MODEL_PATH="$MOUNT_ROOT/models/BAGEL-700M-MoT"
OUTPUT_DIR="$MOUNT_ROOT/wam_runs/bagel/$TIMESTAMP"
PROC_PER_NODE=1

# Finetune BAGLE-A7B
torchrun \
  --nnodes=1 \
  --nproc_per_node=$PROC_PER_NODE \
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
  --num_worker 0 \
  --prefetch_factor 0 \
  --finetune_from_hf True \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --results_dir "$OUTPUT_DIR" \
  --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
  --num_shard $PROC_PER_NODE
