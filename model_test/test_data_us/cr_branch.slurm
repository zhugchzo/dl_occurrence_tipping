#!/bin/bash
#SBATCH -J cr_branch_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o cr_branch.out 
#SBATCH -e cr_branch.err 

python /gs/home/by2109123/test_model/test_data_us/cr_branch_bl.py
python /gs/home/by2109123/test_model/test_data_us/cr_branch_crate.py