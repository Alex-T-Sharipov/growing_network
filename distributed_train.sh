#!/bin/bash
#SBATCH --time 11:59:00
#SBATCH --partition=gpu  # Request 1 GPU
#SBATCH --qos=gpu_free  # Request 1 GPU
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --output=/home/sharipov/monet/output/output_%j.txt
#SBATCH --error=/home/sharipov/monet/output/error_%j.txt

NUM_PROC=$1
shift

# Optionally, set CUDA_VISIBLE_DEVICES if you are sure about the GPU allocation
# CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=$NUM_PROC train.py "$@"
