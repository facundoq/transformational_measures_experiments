#!/usr/bin/env bash


python3 experiment_variance.py -mo "/home/facundoq/variance/models/SimpleConv_mnist_Affine(r=16,s=0,t=0).pt" -me "AnovaFMeasure(ca=sum)" -d "mnist(test,p=0.1)" -t "Affine(r=16,s=0,t=0)" -verbose False -batchsize 64 -num_workers 2


