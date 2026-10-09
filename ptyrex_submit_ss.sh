#!/usr/bin/env bash
#SBATCH --partition=cs05r
#SBATCH --job-name=ptyrex_recon
#SBATCH --nodes 1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node=4
#SBATCH --time 04:00:00
#SBATCH --mem 200G

#SBATCH --constraint=NVIDIA_Blackwell
#SBATCH --error=//dls/science/groups/e02/Chris/data/graphene_600C_defocused/scripts/%j_error.err
#SBATCH --output=//dls/science/groups/e02/Chris/data/graphene_600C_defocused/scripts/%j_output.out
cd /dls_sw/e02/software/PtyREX2026/PtyREX_cuda12/

module load python/cuda12.9

module load hdf5-plugin/1.12

mpirun -np 4 ptyrex_recon -c //dls/science/groups/e02/Chris/Code/Jupyter_Notebooks/Bayes_ptycho_test_final/single_slice_Bayes_optimised.json
