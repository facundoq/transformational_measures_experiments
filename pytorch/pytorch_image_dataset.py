import torch

from torch.utils.data import Dataset

import numpy as np
import transformational_measures as tm
from enum import Enum

class TransformationStrategy(Enum):
    random_sample="random_sample"
    iterate_all="iterate_all"

    def samples(self,n_samples,n_transformations):
        if self == TransformationStrategy.random_sample:
            return n_samples
        elif self.transformation_strategy == TransformationStrategy.iterate_all:
            return n_samples * n_transformations
        else:
            raise ValueError(f"Unsupported TransformationStrategy {self}")

    def get_index(self,idx,n_samples,n_transformations):
        if self == TransformationStrategy.iterate_all:
            i_sample = idx % self.n_samples
            i_transformation = idx // n_samples
        else: # self == TransformationStrategy.random_sample:
            i_sample = idx
            i_transformation = np.random.randint(0, n_transformations)
        return i_sample, i_transformation
    
    def get_indices(self, idx,n_samples,n_transformations):
        if self == TransformationStrategy.iterate_all:
            i_sample = [i % self.n_samples for i in idx]
            i_transformation = [i // n_samples for i in idx]
        else: # self == TransformationStrategy.random_sample:
            i_sample = idx
            i_transformation = np.random.randint(0, n_transformations, size=(len(idx),))
        return i_sample, i_transformation




class ImageDataset(Dataset):
    def __init__(self, image_dataset:Dataset, transformations:tm.TransformationSet=None, transformation_scheme:TransformationStrategy=None,normalize=False):

        if transformation_scheme is None:
            transformation_scheme = TransformationStrategy.random_sample
        self.transformation_strategy = transformation_scheme

        self.dataset=image_dataset

        if transformations is None:
            self.transformations=[tm.IdentityTransformation()]
        else:
            self.transformations=transformations
        self.n_transformations=len(self.transformations)
        self.n_samples = len(self.dataset)



    def __len__(self):
        return self.transformation_strategy.samples(self.n_samples,self.n_transformations)

class ImageClassificationDataset(ImageDataset):
    def __getitem__(self,idx):
        i_sample,i_transformation=self.transformation_strategy.get_index(idx,self.n_samples,self.n_transformations)
        x,y = self.dataset[i_sample]
        t = self.transformations[i_transformation]
        x= t(x.float())
        y=y.type(dtype=torch.LongTensor)
        return x, y[0]


class ImageTransformRegressionDataset(ImageDataset):
    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        i_sample,i_transformation=self.transformation_strategy.get_index(idx,self.n_samples,self.n_transformations)
        # print(self.dataset)
        s = self.dataset[i_sample]
        # print(s.shape)
        t = self.transformations[i_transformation]
        ts = t(s.float())
        # print(t.parameters())
        return ts,t.parameters().float()

class ImageTransformRegressionNormalizedDataset(ImageDataset):

    def __init__(self, image_dataset: Dataset, transformations: tm.TransformationSet = None,
                 transformation_scheme: TransformationStrategy = None, normalize=False):
        super().__init__(image_dataset,transformations,transformation_scheme,normalize)
        self.min,self.max=self.transformations.parameter_range()
        self.delta = self.max - self.min

    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        i_sample,i_transformation=self.transformation_strategy.get_index(idx,self.n_samples,self.n_transformations)
        # print(self.dataset)
        # print(self.dataset[i_sample])
        s = self.dataset[i_sample]
        # print(s.shape)
        t = self.transformations[i_transformation]
        # print(s.shape,s.dtype)
        ts = t(s.float())
        # print(t.parameters())
        target = t.parameters().float()
        # old_target=target.clone()
        target -= self.min
        target /= self.delta
        # print("old, target,mi,ma", old_target,target, mi, ma)
        return ts,target