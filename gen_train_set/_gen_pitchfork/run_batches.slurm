#!/bin/bash
#SBATCH -J dataset_pitchfork
#SBATCH -p cpu-high
#SBATCH -N 1 
#SBATCH --cpus-per-task=1 
#SBATCH -t 168:00:00 
#SBATCH -o dataset_pitchfork.out 
#SBATCH -e dataset_pitchfork.err 
#SBATCH -a 1-50

# Get command line parameters
batch_num=$SLURM_ARRAY_TASK_ID # Batch number (we send each batch to a different CPUs to run in parallel)
bif_max=1000

# Make batch-specific directory for storing output
mkdir -p output/batch$batch_num/sup_pitchfork/output_sims
mkdir -p output/batch$batch_num/sub_pitchfork/output_sims

cd output/batch$batch_num

# Set up timer
start=`date +%s`

echo Job released

# Generate a model and output equi.csv, pars.csv
echo Run gen_model.py
python3 /gen_training_data.py $batch_num $bif_max

# Convert label data and split into training, test, validation
echo "Convert data to correct form for training"
python3 /to_traindata.py $bif_max $batch_num

# Remove single value files
rm sup_pitchfork/value*
rm sub_pitchfork/value*

# End timer
end=`date +%s`
runtime=$((end-start))
echo "Job successfully finished in time of $runtime" seconds.