#!/bin/bash
#SBATCH -J combine_forward
#SBATCH -p cpu-low
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o combine_forward.out
#SBATCH -e combine_forward.err


# Command line arguments
num_batches=50 # number of batches generated
ts_len=500 # time series length


# Run Python file to stack the label and group data
/gen_train_set_set/stack_labels_groups.py $num_batches $ts_len

# Move time series data from batches to combined directory
mkdir -p output_forward/ts_$ts_len/hopf/combined/output_resids
mkdir -p output_forward/ts_$ts_len/fold/combined/output_resids
mkdir -p output_forward/ts_$ts_len/branch/combined/output_resids

let min=1
let max=num_batches

for i in $(seq $min $max)
do
   cp output_forward/ts_$ts_len/batch$i/hopf/output_resids/* output_forward/ts_$ts_len/hopf/combined/output_resids
   cp output_forward/ts_$ts_len/batch$i/fold/output_resids/* output_forward/ts_$ts_len/fold/combined/output_resids
   cp output_forward/ts_$ts_len/batch$i/branch/output_resids/* output_forward/ts_$ts_len/branch/combined/output_resids
done

# Zip the folders
cd output_forward/ts_$ts_len/fold/combined
zip -r output_resids.zip output_resids
cd ../..
cd hopf/combined
zip -r output_resids.zip output_resids
cd ../..
cd branch/combined
zip -r output_resids.zip output_resids


cd ../../../