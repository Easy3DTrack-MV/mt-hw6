#!/bin/bash
#SBATCH -J de   # Job name
#SBATCH -o Slurm_out/default.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e Slurm_out/default.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 16   # Total number of CPU cores requrested
#SBATCH --mem=120gb    # CPU Memory pool for all cores
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH --mail-user=tli79@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu --gres=gpu:a6000:1   # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

nvidia-smi

cd /share/kuleshov/jy928/mt/hw && conda run -p /share/kuleshov/jy928/anaconda3/envs/mt-default --no-capture-output python default.py