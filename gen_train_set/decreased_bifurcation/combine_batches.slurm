#!/bin/bash
#SBATCH -J combine_reverse
#SBATCH -p cpu-low
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o combine_reverse.out
#SBATCH -e combine_reverse.err


# Command line arguments
num_batches=25 # number of batches generated
ts_len=500 # time series length


# Run Python file to stack the label and group data
/gs/home/by2109123/anaconda3/envs/comp/bin/python3 /gs/home/by2109123/gen_train_set/decreased_bifurcation/stack_labels_groups.py $num_batches $ts_len

# Move time series data from batches to combined directory
mkdir -p output/ts_$ts_len/hopf/combined/output_resids
mkdir -p output/ts_$ts_len/fold/combined/output_resids
mkdir -p output/ts_$ts_len/branch/combined/output_resids

let min=1
let max=num_batches

for i in $(seq $min $max)
do
   cp output/ts_$ts_len/batch$i/hopf/output_resids/* output/ts_$ts_len/hopf/combined/output_resids
   cp output/ts_$ts_len/batch$i/fold/output_resids/* output/ts_$ts_len/fold/combined/output_resids
   cp output/ts_$ts_len/batch$i/branch/output_resids/* output/ts_$ts_len/branch/combined/output_resids
done

# Zip the folders
cd output/ts_$ts_len/fold/combined
zip -r output_resids.zip output_resids
cd ../..
cd hopf/combined
zip -r output_resids.zip output_resids
cd ../..
cd branch/combined
zip -r output_resids.zip output_resids


cd ../../../