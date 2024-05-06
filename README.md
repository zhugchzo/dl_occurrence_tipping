# Deep learning for the occurrence of tipping points

This is the code repository to accompnay the article:

***Deep learning for the occurrence of tipping points.*** *Chengzuo Zhuge, Jiawei Li, Wei Chen.*

# Requirements

Python 3.7 is required. To install python package dependencies, use the command

```
pip install -r requirements.txt
```

within a new virtual environment.

The bifurcation continuation software AUTO-07P is required. Installation instructions are provided at http://www.macs.hw.ac.uk/~gabriel/auto07/auto.html

# Directories

**./dl_model:** Code to train the deep learning algorithm. And trained deep learning models used in manuscript.

**./dl_null_model:** Code to train the null model. And trained null models used in manuscript.

**./dl_combined_model:** Code to train the combined model. And trained combined models used in manuscript.

**./gen_train_set:** Code to generate training data for the deep learning algorithm.

**./model_test:** Code to simulate, perform regular and irregular sampling, and compute residual time series for the test models, to compute indicators of competing algorithms in the regularly-sampled situation, and to test trained deep learning models on simulated model test time series. These include the May's harvesting model, three-species food chain model, consumer−resource model, global energy balance model, middle Pleistocene transition model, Amazon rainforest dieback model, sleep-wake hysteresis loop model and Sprott B hysteresis bursting model. The model test time series data after sampling and computing residual we used in manuscript is also in this directory.

**./empirical_test:** Code to perform irregular sampling and compute residual time series for the empirical datasets, and to test trained deep learning models on empirical test time series. These include the cyanobacteria microcosm experiment, the thermoacoustic experiment and the metallic elements in sediment cores. The original empirical datasets and the empirical test time series data after sampling and computing residual we used in manuscript are also in this directory.

**./results:** Experimental results of the trained deep learning models on the model time series data and the empirical test time series data.

**./draw_fig:** Code to generate figures used in manuscript.

**./figures:** Figures used in manuscript.

# Workflow

The results in the paper are obtained from the following workflow:

## Generate the training data. 
We generate six sets of training data categorized by three bifurcation types and two noise types for six deep learning models. Each deep learning model is trained on 200,000 time series (100,000 time series with bifurcation parameter increasing and 100,000 time series with bifurcation parameter decreasing) with length of 500 data points. Run

```
sbatch run_batches_forward.slurm
sbatch run_batches_reverse.slurm
```

where the former is used to generate time series with increasing bifurcation parameter, while the latter is used to generate time series with decreasing bifurcation parameter. We run 50 batches in parallel for both of them on a CPU cluster at the Beihang University. One batch generates 12000 time series, consisting of 2000 time series for each bifurcation type and noise type (fold-white, Hopf-white, transcritical-white, fold-red, Hopf-red, transcritical-red). Each time series is saved as a csv file. A total of 1,200,000 (2x50x12000) time series are generated for training six deep learning models.

Once every batch has been generated, the output data from each batch is combined using

```
sbatch combine_batches_forward.slurm
sbatch combine_batches_reverse.slurm
```

This also stacks the labels.csv and groups.csv files, and compresses the folder containing the time series data. The final compressed output comes out at GB with increasing bifurcation parameter and GB with decreasing bifurcation parameter. Training datasets we used in manuscript are archived on Zenodo at https://.

## Train the deep learning algorithm.


























