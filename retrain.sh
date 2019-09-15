#!/usr/bin/env bash


./experiment_training.py -m "ResNet" -d "cifar10" -transformation "Affine(r=0,s=0,t=0)"
#./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=0,t=4)"
#./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=2,s=0,t=0)"
#./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=1,t=0)"


./experiment_training.py -m "FFNet" -d "mnist" -transformation "Affine(r=0,s=0,t=0)"