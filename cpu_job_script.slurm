#!/bin/bash

#SBATCH --job-name=mass_inf
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool2
#SBATCH --nodes=1

module load python/3.10.8-gpu 
module load pytorch/1.13.1-gpu

export PATH=/work/tc062/shared/bin:$PATH

source /work/tc062/tc062/s2501147/venv/mgpu_env/bin/activate

echo "Starting job on $(date)"

srun python mp3_encoder.py

echo "Finished job on $(date)"