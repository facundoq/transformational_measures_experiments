import torch
import os
import numpy as np
import transformational_measures as tm
from typing import Generator,Any,Tuple,List
from nptyping import Array

ndarray=Array[np.float]

SingleActivationsGenerator=Generator[Tuple[ndarray,List[ndarray]],None,None]
ActivationsGenerator=Generator[Tuple[tm.Transformation,SingleActivationsGenerator],None,None]

class FakeActivationsIterator(tm.ActivationsIterator):
    def __init__(self,batch_size=6):
        self.data = [[7, 4, 6],
                     [9, 3, 1],
                     [5, 6, 3],
                     [8, 2, 5],
                     [6, 7, 3],
                     [8, 5, 4],
                     [6, 5, 6],
                     [10, 4, 5],
                     [7, 1, 7],
                     [4, 3, 3],
                     ]
        self.data = np.array(self.data)
        self.batch_size=batch_size

    def transformations_first(self)->ActivationsGenerator:
        n,t=self.data.shape
        for i in range(t):
            yield i,self.batch_results_transform(i)

    def batch_results_transform(self,i:int)->SingleActivationsGenerator:
        n, t = self.data.shape
        yield np.zeros(n),[self.data[:,i]]


    def samples_first(self)->ActivationsGenerator:
        n,t=self.data.shape
        for i in range(n):
            yield i,self.batch_results_sample(i)


    def batch_results_sample(self,i:int)->SingleActivationsGenerator:
        n, t = self.data.shape
        yield np.zeros(t),[self.data[i,:]]

    def layer_names(self) -> [str]:
        return ["x"]

activations_iterator=FakeActivationsIterator()

# print("transformations_first")
# for t,samples_iterator in activations_iterator.transformations_first():
#     print(t)
#     for transformed,activations in samples_iterator:
#         print(transformed,activations)
#
#
# print("samples_first")
# for samples,transformations_iterator in activations_iterator.samples_first():
#     print(samples)
#     for transformed,activations in transformations_iterator:
#         print(transformed,activations)

m=tm.AnovaFMeasure(conv_aggregation=tm.ConvAggregation.none)
mr=m.eval(activations_iterator)
assert (mr.layers[0]-8.18)<0.001
print(mr.layers)

ma=tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none,alpha=0.95)
mra=ma.eval(activations_iterator)
assert mra.layers[0]
print(mra.layers)