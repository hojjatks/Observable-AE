#!/bin/bash

# Load the Conda module
# Activate your Conda environment

# Submit this script with: sbatch thefilename
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH --partition=gpu
#SBATCH -J "autoencoder"   # job name
#SBATCH --mail-user=hkaveh@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
conda init
conda activate autoencoder

# Run your Python script
python Observable-AE-noaddnet.py