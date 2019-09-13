#!/usr/bin/env bash


./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=0,t=3)"
./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=0,t=4)"
./experiment_training.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=1,t=0)"
