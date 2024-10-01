#!/bin/bash
#SBATCH -J food_hopf_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o food_hopf.out 
#SBATCH -e food_hopf.err 

python /gs/home/by2109123/test_model/test_data_us/food_hopf_bl.py
python /gs/home/by2109123/test_model/test_data_us/food_hopf_crate.py