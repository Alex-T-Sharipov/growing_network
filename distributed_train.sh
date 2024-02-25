#!/bin/bash

# Activate the virtual environment
source /Users/alexanders/LIONS_Semester_Project/.venv/bin/activate

# Get the number of processes
NUM_PROC=$1
shift

# Use the 'which' command to ensure that we're using the 'torchrun' from the virtual environment
TORCHRUN=$(which torchrun)

# Check if 'torchrun' is available in the virtual environment
if [ -z "$TORCHRUN" ]; then
  echo "torchrun not found in the virtual environment"
  exit 1
fi

# Set CUDA visible devices
CUDA_VISIBLE_DEVICES=2,3,4

# Run 'torchrun' with the virtual environment's Python
$TORCHRUN --nproc_per_node=$NUM_PROC train.py "$@"

# Deactivate the virtual environment
deactivate