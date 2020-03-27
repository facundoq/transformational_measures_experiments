# Measuring rotational (in)variance in Convolutional Neural Networks

This repository contains the code necessary to obtain the experimental results published in the article [Measuring (in)variances to transformations in Neural Networks]() (link and bib entry coming soon).

## Abstract

`Coming soon` 

## What can you do with this code

You can train a model on the datasets:
* [MNIST](http://yann.lecun.com/exdb/mnist/) 
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) 

You can choose a set of transformations as data augmentation. Currently supported:
* Affine Transformations
    * Rotation
    * Translation
    * Scaling
    * ... and combinations

Afterwards, you can also calculate various measures of invariance/equivariance for those models, and produce several plots/visualizations.

The available models are: 
* [Resnet](https://arxiv.org/abs/1512.03385)
* [VGG16](https://arxiv.org/abs/1409.1556)
* [AllConvolutional network](https://arxiv.org/abs/1412.6806)
* [Simple Convolutional Network](https://github.com/facundoq/rotational_invariance_data_augmentation/blob/master/pytorch/model/simple_conv.py)  

 


## How to run

These instructions have been tested on a modern ubuntu-based (18.10/19.10) distro with python version>=3.5. All experiments are based on models defined and trained with PyTorch.  

1. Clone the repository and cd to it:
    * `git clone https://github.com/facundoq/transformational_measures_experiments.git`
    * `cd transformational_measures_experiments` 
2. Create a virtual environment and activate it (requires python3 with the venv module and pip):
    * `python3 -m venv .env`
    * `source .env/bin/activate`
3. Update `pip`, `wheel` and `setuptools` to the latest versions:
    *  pip install --upgrade pip wheel setuptools
4. Install libraries 
    * `pip install -r requirements.txt`
    * Also install a distribution of Pytorch (requirements-torch.txt is the one we used for cuda 10 but you should the appropriate version for your hardward/drivers/cuda).
5. (optional) Install the Computer Modern fonts if you want to recreate the figures as in the paper.
    1. `sudo apt install cm-super` in ubuntu
6. Run the experiments:
    * `train.py` trains individual models with a given dataset, and set of transformation as data augmentation. Models are stored by default in `~/variance/models` and a plot of their loss/accuracy history while training can be found in `~/variance/training_plots`
    * `measure.py`  calculates an invariance/equivariance measure for a given model /dataset/transformation/measure combination. Results are saved by default to `~/variance/results/`
    * `run.py` uses the previous scripts to generate the plots and analysis found in the paper. Classes that subclass `Experiment` can be run independently or as a group when executing `experiment.py`. Results are stored by default in `~/variance/plots/`. Each `Experiment` subclass contains a plaintext `description` field with a briefly summary of its goal and methods.
        * Execute run.py -e <experiment name>, where <experiment name> is one of the experiments defined in run.py
