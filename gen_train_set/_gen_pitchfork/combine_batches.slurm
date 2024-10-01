#!/bin/bash
#SBATCH -J combine_pitchfork
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o combine_pitchfork.out
#SBATCH -e combine_pitchfork.err

# Command line arguments
num_batches=50 # number of batches generated

# Run Python file to stack the label and group data
python3 /gen_pitchfork/stack_labels_groups.py $num_batches

# Move time series data from batches to combined directory
mkdir -p output/sup_pitchfork/combined/output_sims
mkdir -p output/sub_pitchfork/combined/output_sims

let min=1
let max=num_batches

for i in $(seq $min $max)
do
   cp output/batch$i/sup_pitchfork/output_sims/sup_tseries* output/sup_pitchfork/combined/output_sims
   cp output/batch$i/sub_pitchfork/output_sims/sub_tseries* output/sub_pitchfork/combined/output_sims
done

# Zip the folders
cd output/sup_pitchfork/combined
zip -r output_sims.zip output_sims
cd ../../..
cd output/sub_pitchfork/combined
zip -r output_sims.zip output_sims