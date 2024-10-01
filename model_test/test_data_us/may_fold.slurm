#!/bin/bash
#SBATCH -J may_fold_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o may_fold.out 
#SBATCH -e may_fold.err 

python /gs/home/by2109123/test_model/test_data_us/may_fold_bl.py
python /gs/home/by2109123/test_model/test_data_us/may_fold_crate.py