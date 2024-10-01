#!/bin/bash
#SBATCH -J MPT_hopf_us
#SBATCH -p cpu-high 
#SBATCH -N 1 
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 
#SBATCH -o MPT_hopf.out 
#SBATCH -e MPT_hopf.err 

python /gs/home/by2109123/test_model/test_data_us/MPT_hopf_bl.py
python /gs/home/by2109123/test_model/test_data_us/MPT_hopf_crate.py