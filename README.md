# Measuring rotational (in)variance in Convolutional Neural Networks

This repository contains the code necessary to obtain the experimental results published in the article [Measuring (in)variances to transformations in Neural Networks]() (link and bib entry coming soon).

## Abstract

``
  Invariances in neural networks are useful and necessary for many tasks. Currently, there are various proposals on how to achieve invariance, based on both specialized model design and data augmentation. However, the actual representation of the invariance, especially for models trained with data augmentation, has received little attention. We define efficient and interpretable measures to quantify the invariance of neural networks in terms of their internal representation. The measures can be applied to any neural network model, but can also be specialized for specific modules such as convolutional layers. Through various experiments, we validate the measures and their properties in the domain of affine transformations and the CIFAR10 and MNIST datasets, including their stability, their relation to sample and transformation size, and interpretability.
  ``

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

These instructions have been tested on a modern ubuntu-based (22.04) distro with python version>=3.10. All experiments are based on models defined and trained with PyTorch.  

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
    * Also install a distribution of Pytorch (requirements-torch.txt is the one we used for cuda 10 but you should the appropriate version for your hardware/drivers/cuda).
5. (optional) Install the Computer Modern fonts if you want to recreate the figures as in the paper.
    1. `sudo apt install cm-super` in ubuntu
6. Run the experiments:
    * `./invariance_run.py` Executes invariance experiments (see folder `experiments/invariance/` for their individual description)
    * `./same_equivariance_run.py` Executes same-equivariance experiments (see folder `experiments/same_equivariance/` for their individual description)
