#!/bin/bash
#SBATCH --job-name=mallm1
#SBATCH --time=72:00:00
#SBATCH -N 1

source ~/.bashrc
conda activate mallm

# run
python3 batch_mallm.py exp2/xsum_batch.json
python3 batch_mallm.py exp2/etpc_batch.json
python3 batch_mallm.py exp2/simple_ethical_questions_batch.json
python3 batch_mallm.py exp2/strategyqa_batch.json
python3 batch_mallm.py exp2/wmt_batch.json
python3 batch_mallm.py exp2/squad_v2_batch.json
