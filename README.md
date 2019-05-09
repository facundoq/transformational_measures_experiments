# Measuring rotational (in)variance in Convolutional Neural Networks

This repository contains the code necessary to obtain the experimental results published in the article [Measuring (in)variances to transformations in Neural Networks]() (link and bib entry coming soon).

## Abstract

`Coming soon` 

## What can you do with this code

You can train a model on the [MNIST](http://yann.lecun.com/exdb/mnist/) or [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. One model will be trained with a vanilla version of the dataset. Then, for each set of transformations desired, an additional model will be trained. For example, for rotations a *rotated* model is trained, for which the dataset's samples were randomly rotated.

The available models are [Resnet](), [VGG16](), [AllConvolutional network](https://arxiv.org/abs/1412.6806) and a [simple Convolutional Network](https://github.com/facundoq/rotational_invariance_data_augmentation/blob/master/pytorch/model/simple_conv.py)  

Afterwards, you can measure the (in)variance of each activation of the networks for the different transformations, and visualize them as heatmaps or plots. 


## How to run

These instructions have been tested on a modern ubuntu-based distro with python version>=3.5.  

* Clone the repository and cd to it:
    * `git clone https://github.com/facundoq/variance_measure.git`
    * `cd variance_measure` 
* Create a virtual environment and activate it (requires python3 with the venv module and pip):
    * `python3 -m venv .env`
    * `source .env/bin/activate`
* Update `pip`, `wheel` and `setuptools` to the latest versions:
    *  pip install --upgrade pip wheel setuptools
* Install libraries 
    * `pip install -r requirements.txt`
    
* Run the experiments with `python experiment> <model> <dataset>`
    * `experiment_train.py` trains models with the dataset: one with the vanilla version, the other with a data-augmented version via the desired transformations.
    * `experiment_variance.py`  calculates the variance of the activations of the model for the model/dataset combinations, with vanilla and transformed versions. Results are saved by default to `~/variance_results/`
    * `plot_variances_models.py` generates plots of the variances for each/model dataset combination found in `~/variance_results/`. Both stratified/non-stratified versions of the measure are included in the plots. 
    
* The folder `plots` contains the results for any given model/dataset combination

