#!/bin/bash
#SBATCH -J dataset_forward
#SBATCH -p cpu-high
#SBATCH -N 1 
#SBATCH --cpus-per-task=1 
#SBATCH -t 168:00:00 
#SBATCH -o dataset_forward.out 
#SBATCH -e dataset_forward.err 
#SBATCH -a 1-50

# Get command line parameters
batch_num=$SLURM_ARRAY_TASK_ID # Batch number (we send each batch to a different CPUs to run in parallel)
ts_len=500 # Length of time series to generate for training data

# Other parameters
bif_max=2000 # Number of each type of bifurcation requested (we do 2000 per batch)


# Make batch-specific directory for storing output_forward
mkdir -p output_forward/ts_$ts_len/batch$batch_num/hopf
mkdir -p output_forward/ts_$ts_len/batch$batch_num/fold
mkdir -p output_forward/ts_$ts_len/batch$batch_num/branch
# Move to the batch-specific directory
cd output_forward/ts_$ts_len/batch$batch_num

# Set up timer
start=`date +%s`

echo Job released

# Define variables that count numbers of bifurcations

if [-e output_forward_counts/list_counts.txt]
then
    hopf_count=$(sed '1q;d' output_forward_counts/list_counts.txt)
    fold_count=$(sed '2q;d' output_forward_counts/list_counts.txt)
    branch_count=$(sed '3q;d' output_forward_counts/list_counts.txt)
else
    hopf_count=0
    fold_count=0
    branch_count=0   
fi



# While loop
while [ $hopf_count -lt $bif_max ] || [ $fold_count -lt $bif_max ] || [ $branch_count -lt $bif_max ]
do
printf "\nBifurcation counts\n"
printf "hopf_count = $hopf_count, fold_count = $fold_count, branch_count = $branch_count"
printf "\n\n"


# Generate a model and output_forward equi.csv, pars.csv
echo Run gen_model.py
/gen_train/gen_model.py

# Copy AUTO model and constants files to batch-specific directory
cp /gen_train/c_forward.model c_forward.model
cp /gen_train/model.f90 model.f90

# Run bifurcation continuation using AUTO and output_forward b.out files for each varied parameter
# (Make sure AUTO runs using Python 2)
echo Run run_cont.auto
# This should not take more than 10 mins - if it does, cancel the run and create new model
timeout 600 auto /gen_train/run_cont.auto
if [[ $? == 124 ]]
then
    echo AUTO cancelled as it took over 10 minutes
	continue
fi
# Remove unnecessary files
rm -f c_forward.model
rm -f model.f90

# For each parameter with a bifurcation, run simulations up to the
# bifurcation point, and output_forward 500 points prior to the transition.
# Also run a null case where parameters are fixed.
echo Run stoch_sims.py
/gen_train/stoch_sims.py $hopf_count $fold_count $branch_count $bif_max $batch_num $ts_len


# Update counting variables
hopf_count=$(sed '1q;d' output_forward_counts/list_counts.txt)
fold_count=$(sed '2q;d' output_forward_counts/list_counts.txt)
branch_count=$(sed '3q;d' output_forward_counts/list_counts.txt)


# Remove old model, AUTO and simulation files to save space
rm -r output_forward_model
rm -r output_forward_auto

done



# Convert label data and split into training, test, validation
echo "Convert data to correct form for training"
/gen_train/to_traindata.py $bif_max $batch_num

# Remove single value files
rm hopf/output_forward_values/value*
rm fold/output_forward_values/value*
rm branch/output_forward_values/value*

# Compute residual dynamics after Lowess smoothing for each time series
echo "Compute residuals"
/gen_train/compute_resids.py $bif_max $batch_num


# End timer
end=`date +%s`
runtime=$((end-start))
echo "Job successfully finished in time of $runtime" seconds.

# Change back to original directory
cd /gen_train/


