#!/bin/bash

#SBATCH --job-name=recon130000
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load python/3.10.8-gpu 
module load pytorch/1.13.1-gpu

source /work/tc062/tc062/s2501147/venv/mgpu_env/bin/activate

srun python variable_length_restauration.py --epochs 10
