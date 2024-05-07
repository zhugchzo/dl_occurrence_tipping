# Deep learning for the occurrence of tipping points

This is the code repository to accompnay the article:

***Deep learning for the occurrence of tipping points.*** *Chengzuo Zhuge, Jiawei Li, Wei Chen.*

# Requirements

Python 3.7 is required. To install python package dependencies, use the command

``` setup
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

1. **Generate the training data.** We generate six sets of training data categorized by three bifurcation types and two noise types for six deep learning models. Each deep learning model is trained on 200,000 time series (100,000 time series with bifurcation parameter increasing and 100,000 time series with bifurcation parameter decreasing) with length of 500 data points. Run

   ```bash
   sbatch gen_train_set/run_batches_forward.slurm
   sbatch gen_train_set/run_batches_reverse.slurm
   ```

   where the former is used to generate time series with increasing bifurcation parameter, while the latter is used to generate time series with decreasing bifurcation parameter. We run 50 batches in parallel for both of them on a CPU cluster at the Beihang University. One batch generates 12000 time series, consisting of 2000 time series for each bifurcation type and noise type (fold-white, Hopf-white, transcritical-white, fold-red, Hopf-red, transcritical-red). Each time series is saved as a csv file. A total of 1,200,000 (2x50x12000) time series are generated for training six deep learning models.

   Once every batch has been generated, the output data from each batch is combined using

   ```bash
   sbatch gen_train_set/combine_batches_forward.slurm
   sbatch gen_train_set/combine_batches_reverse.slurm
    ```

    This also stacks the labels.csv and groups.csv files, and compresses the folder containing the time series data. The final compressed output comes out at GB with increasing bifurcation parameter and GB with decreasing bifurcation parameter. Training datasets we used in manuscript are archived on Zenodo at https://.

2. **Train the deep learning algorithm.** We train five neural networks on each training set and report the performance of the model averaged over them. For example, to train a single neural network on the fold-white training set of a index kk, run

   ```python
   python ./dl_model/DL_training_fold_white.py $kk
   ```

   This will export the trained model (including weights, biases and architecture) to the directory `./dl_model/`. We run this for kk in [1,2,...,5]. Taking kk as command line parameters allows training of multiple neural networks in parallel if one has access to mulitple CPUs. Time to train a single neural network using a CPU is approximately 72 hours. The same process can be used to train both the null model and the combined model as comparisons for the deep learning algorithm in manuscript, with the corresponding code located in the directory `./dl_null_model/` and `./dl_combined_model/` respectively.
 
3. **Generate model time series for testing.** Simulate the model time series used to test the DL algorithm. Code to do this is available in `./model_test/`. For example, to simulate trajectories of May's harvesting model going through a fold bifurcation, and perform regular and irregular sampling for the trajectories, run

   ```python
   python ./model_test/sim_data/sim_may_fold.py
   ```

   The same file notation is used for the other models except sleep-wake hysteresis loop model and Sprott B hysteresis bursting model. These scripts also detrend the time series to get residuals, and compute indicators of competing algorithms in the regularly-sampled situation.

   Simulate the model time series with hysteresis phenomenon, code to do this is available in `./model_test/sim_hysteresis/`. For example, to simulate trajectories of sleep-wake hysteresis loop model, run 

   ```python
   python ./model_test/sim_hysteresis/sim_sleep-wake_original_series.py
   ```

   , then perform irregular sampling for the trajectories, run

   ```python
   python ./model_test/sim_hysteresis/sim_sleep-wake.py
   ```

   The same file notation is used for the Sprott B hysteresis bursting model. These scripts also detrend the time series to get residuals.

4. **Process empirical data.** Scripts to process the empirical data (including irregular sampling and computing residuals) are availalble in the directory `./empirical_test/sim_data_nus`.

5. **Generate predictions by deep learning models.** Before we feed the residual and parameter data into the neural networks to make predictions, there are two preprocessing matters. First, we pad both residual and parameter series are padded on the left by zeroes to a length of 500. Second,  each residual time series is normalized by dividing each time series data point by the average absolute value of the residuals across the entire time series. In addition, each parameter time series is also normalized: each data point in the time series is subtracted by the initial value of the parameter series, and then divided by the distance between the initial and final values of the parameter series. These two preprocessing matters are also performed for training data before they are fed into the neural networks in the **Generate the training data**.

   For predictions of model time series, the code to test on regularly-sampled model time series and irregularly-sampled model time series are availalble in the directory `./model_test/test_data_us/` and `./model_test/test_data_nus/` respectively. For example, to apply the deep learning algorithm on regularly-sampled time series generated from May's harvesting model with 11 different initial values of the bifurcation parameter, run

   ```python
   python ./model_test/test_data_us/may_fold.py
   ```

   The code for generating predictions using competing algorithms is also included in `test_data_us/may_fold.py`. The same file notation is used for the time series generated from other models. Furthermore, in the situation of irregularly-sampled time series generated from May's harvesting model, run

   ```python
   python ./model_test/test_data_nus/may_fold.py
   ```

   The code for generating predictions by null model and combined model is also included in `test_data_nus/may_fold.py`. The same file notation is used for the time series generated from other models.

   For predictions of empirical time series, to test on all irregularly-sampled empirical time series used in manuscript, run

   ```python
   python ./empirical_test/test_empirical_data/empirical_sum.py
   ```

   **If using your own data, it is important to detrend it using a Lowess filter with span 0.20 and perform the two preprocessing matters mentioned above.**

# Data sources

The empirical data used in this study are available from the following sources:

1. **cyanobacterial population collapse** data is availalble in the text file `./empirical_test/empirical_data/microcosm.txt`. Data was collected by AJ Veraart et al. and was first published in [Veraart A J, Faassen E J, Dakos V, et al. Recovery rates reflect distance to a tipping point in a living system[J]. Nature, 2012, 481(7381): 357-359.](https://www.nature.com/articles/nature10723)

2. **Thermoacoustic instability** data is availalble in the text files `./empirical_test/empirical_data/thermoacoustic_20mv.txt`, `./empirical_test/empirical_data/thermoacoustic_40mv.txt` and `./empirical_test/empirical_data/thermoacoustic_60mv.txt`. Data was collected by Induja Pavithran and R. I. Sujith and was first published in [Pavithran I, Sujith R I. Effect of rate of change of parameter on early warning signals for critical transitions[J]. Chaos: An Interdisciplinary Journal of Nonlinear Science, 2021, 31(1).](https://pubs.aip.org/aip/cha/article-abstract/31/1/013116/1059628/Effect-of-rate-of-change-of-parameter-on-early?redirectedFrom=fulltext)

3. **Sedimentary archive** data is availalble in the text files `./empirical_test/empirical_data/hypoxia_64PE_Mo_fold.txt` and `./empirical_test/empirical_data/hypoxia_64PE_U_branch.txt`. Data was collected from the Mediterranean Sea and is available at the [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.923197) data repository.




























