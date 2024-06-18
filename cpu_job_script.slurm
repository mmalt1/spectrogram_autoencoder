#!/bin/bash

#SBATCH --job-name=autoenc_pics
#SBATCH --time=00:15:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool2
#SBATCH --nodes=1

module load python/3.10.8-gpu 
module load pytorch/1.13.1-gpu

source /work/tc062/tc062/s2501147/venv/mgpu_env/bin/activate

srun autoencoder/autoencoder_digit_user.py