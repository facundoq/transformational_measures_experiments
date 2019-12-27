from .activations_iterator import ActivationsIterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from .pytorch_image_dataset import ImageDataset
import numpy as np

from transformation_measure import TransformationSet
from transformation_measure.adapters import TransformationAdapter
class PytorchActivationsIterator(ActivationsIterator):

    def __init__(self, model:nn.Module, dataset, transformations:TransformationSet, batch_size=32,num_workers=0,adapter:TransformationAdapter=None,use_cuda=torch.cuda.is_available()):
        '''
        models: a pytorch models that implements the forward_intermediate() method
        dataset: a dataset that yields x,y tuples
        transformations: a list of functions that take a numpy
                        sample and return a transformed one
        '''
        self.model=model
        self.dataset=dataset
        self.transformations=transformations

        self.image_dataset=ImageDataset(self.dataset)
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.adapter=adapter

    def get_transformations(self):
        return self.transformations
    def activation_names(self):
        return self.model.activation_names()

    '''
    Returns the activations of the models by iterating first over transformations and 
    then, for each transformation, over samples
    '''
    def transformations_first(self):
        for transformation in self.transformations:
            dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True,pin_memory=True)
            yield transformation, self.samples_activation(transformation,dataloader)

    def samples_activation(self,transformation,dataloader):
        for batch, _ in dataloader:
            if self.use_cuda:
                batch=batch.cuda()
            batch=self.transform_batch(transformation,batch)
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(batch)
                batch_activations =  [a.cpu().numpy() for a in batch_activations]
                yield batch,batch_activations
    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def samples_first(self):
        dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,drop_last=True)

        with torch.no_grad():
            for batch, y_true in dataloader:
                batch_cpu = batch
                if self.use_cuda:
                    batch = batch.cuda()
                for i in range(batch.shape[0]):
                    x = batch[i, :]
                    yield batch_cpu[i,:], self.transformations_activations(x)

    def transformations_activations(self,x):

        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        for batch in dataloader:
            y, batch_activations = self.model.forward_intermediates(batch)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch,batch_activations


    def transform_sample(self, x: torch.Tensor):

        x = x.unsqueeze(0)
        results=[]
        for i, transformation in enumerate(self.transformations):
            transformed = self.transform_batch(transformation,x)
            results.append(transformed)
        return torch.cat(results)

    def transform_batch(self,transformation,x:torch.Tensor):
        if not self.adapter is None:
            x = self.adapter.pre_adapt(x)
        x = transformation(x)
        if not self.adapter is None:
            x = self.adapter.post_adapt(x)
        return x

from abc import abstractmethod
from torch import nn

class ObservableLayersModule(nn.Module):

    @abstractmethod
    def activation_names(self)->[str]:
        raise NotImplementedError()

    @abstractmethod
    def forward_intermediates(self,args)->(object,[]):
        raise NotImplementedError()

    def n_intermediates(self):
        return len(self.activation_names())
