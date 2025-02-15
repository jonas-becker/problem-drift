#!/bin/bash
#SBATCH --job-name=pd1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH -N 1

source ~/.bashrc
source ~/github/mallm/mallm/bin/activate

# run
python3 batch_mallm.py exp1/exp1_batch.json
