#!/bin/bash

#SBATCH --account=ucb289_asc3
#SBATCH --partition=amem
#SBATCH --job-name=example-job
#SBATCH --output=example-job.%j.out
#SBATCH --time=3:00:00
#SBATCH --qos=mem
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kuad8709@colorado.edu

module purge
module load anaconda
conda activate QST-UGD

# python main.py --POVM "Tetra4" \
#               --n_qubits 6 \
#               --na_state "W_P" \
#               --P_state 0.9 \
#               --ty_state "mixed" \
#               --noise "noise" \
#               --r_path "/scratch/alpine/kuad8709/"

# python helper.py --dir "/scratch/alpine/kuad8709/QST/data/tetra_4/"

python lowmem_main.py
