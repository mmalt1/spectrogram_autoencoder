loss=
for par1 in 15 30 64; do
    sbatch job_script.slurm $par1 $loss aaaa

    