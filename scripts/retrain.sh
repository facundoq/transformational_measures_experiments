#!/usr/bin/env bash


../train.py -m "ResNet" -d "cifar10" -transformation "Affine(r=0,s=0,t=0)"
#./train.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=0,t=4)"
#./train.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=2,s=0,t=0)"
#./train.py -m "AllConvolutional" -d "cifar10" -transformation "Affine(r=0,s=1,t=0)"


../train.py -m "FFNet" -d "mnist" -transformation "Affine(r=0,s=0,t=0)"