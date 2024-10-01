#!/bin/bash
#SBATCH -J global_fold_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o global_fold.out 
#SBATCH -e global_fold.err 

python /gs/home/by2109123/test_model/test_data_us/global_fold_bl.py
python /gs/home/by2109123/test_model/test_data_us/global_fold_crate.py