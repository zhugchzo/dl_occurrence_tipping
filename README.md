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
**./dl_model:** Code to train the deep learning algorithm and trained deep learning models used in our paper.

**./dl_null_model:** Code to train the null model and trained null models used in our paper.

**./dl_combined_model:** Code to train the combined model and trained combined models used in our paper.

**./gen_train_set:** Code to generate training data for the deep learning algorithm.

**./model_test:** Code to simulate and perform regular and irregular sampling for the test models. And code to compute indicators of computing algorithms in the regularly-sampled situation. These include the May's harvesting model, three-species food chain model, consumer−resource model, global energy balance model, middle Pleistocene transition model, Amazon rainforest dieback model, sleep-wake hysteresis loop model and Sprott B hysteresis bursting model.
