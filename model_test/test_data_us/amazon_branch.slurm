#!/bin/bash
#SBATCH -J amazon_branch_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o amazon_branch.out 
#SBATCH -e amazon_branch.err 

python /gs/home/by2109123/test_model/test_data_us/amazon_branch_bl.py
python /gs/home/by2109123/test_model/test_data_us/amazon_branch_crate.py