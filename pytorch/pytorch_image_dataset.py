import torch
#
# from torchvision import transforms
# from torchvision.transforms import functional

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import transformational_measures as tm
from enum import Enum

class TransformationStrategy(Enum):
    random_sample="random_sample"
    iterate_all="iterate_all"

    def samples(self,n_samples,n_transformations):
        if self == TransformationStrategy.random_sample:
            return n_samples
        else:# self.transformation_strategy == TransformationStrategy.iterate_all:
            return n_samples * n_transformations

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
            self.transformations=list(transformations)
        self.n_transformations=len(self.transformations)
        self.n_samples = len(self.dataset)
        # self.normalize=normalize
        # if normalize:
            # self.setup_transformation_pipeline()


    # def setup_transformation_pipeline(self,):
    #     x, y = self.get_all()
    #     n, c, w, h = x.shape
    #     self.h = h
    #     self.w = w
    #     self.c=c
    #     self.mu, self.std = self.calculate_mu_std(x)
    #
    #     # transformations = [transforms.ToPILImage(), ]
    #     #
    #     # transformations.append(transforms.ToTensor())
    #     # transformations.append(transforms.Normalize(mu, std))
    #     # return transforms.Compose(transformations)
    #
    # def calculate_mu_std(self,x:torch.Tensor):
    #
    #     xf = x.float()
    #     dims = (0, 2, 3)
    #
    #     mu = xf.mean(dim=dims,keepdim=True)
    #     std = xf.std(dim=dims,keepdim=True)
    #
    #     std[std == 0] = 1
    #     return mu,std

    def __len__(self):
        return self.transformation_strategy.samples(self.n_samples,self.n_transformations)

    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)

    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        i_sample,i_transformation=self.transformation_strategy.get_index(idx,self.n_samples,self.n_transformations)
        # print(self.dataset)
        s, = self.dataset[i_sample]
        # print(s.shape)
        t = self.transformations[i_transformation]
        s = s.float().unsqueeze(0)
        # print(s.shape,s.dtype)
        ts = t(s).squeeze(0)
        # print(t.parameters())
        return ts,t.parameters().float()
    #
    # def transform_batch(self,x,i_transformation):
    #     x = x.float()
    #     # x = (x-self.mu)/self.std
    #     for i in range(x.shape[0]):
    #         sample=x[i:i+1,:]
    #         print(i_transformation)
    #         t = self.transformations[i_transformation[i]]
    #         x[i,:] = t(sample)
    #     return x

    def get_indices(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return self.transformation_strategy.get_indices(idx,self.n_samples,self.n_transformations)


class ImageClassificationDataset(ImageDataset):
    pass
    # def get_batch(self,idx):
    #     i_sample,i_transformation = self.get_indices(idx)
    #
    #     x,y=self.dataset.get_batch(i_sample)
    #     x=self.transform_batch(x,i_transformation)
    #
    #     y=y.type(dtype=torch.LongTensor)
    #     return x, y


class ImageTransformRegressionDataset(ImageDataset):
    pass
    #
    # def get_batch(self, idx):
    #     i_sample, i_transformation = self.get_indices(idx)
    #     print(f"getting batch {idx}, {i_sample} {i_transformation}")
    #     x, = self.dataset.get_batch(i_sample)
    #     x = self.transform_batch(x, i_transformation)
    #     t = self.transformations[i_transformation]
    #     y = t.parameters()
    #     return x, y
