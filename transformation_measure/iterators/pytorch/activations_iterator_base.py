from transformation_measure.iterators.activations_iterator import ActivationsIterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformation_measure.iterators.pytorch_image_dataset import ImageDataset

from transformation_measure import TransformationSet,Transformation
from transformation_measure.adapters import TransformationAdapter

from abc import abstractmethod
from torch import nn
import transformation_measure as tm


class ObservableLayersModule(nn.Module):

    @abstractmethod
    def activation_names(self) -> [str]:
        raise NotImplementedError()

    @abstractmethod
    def forward_intermediates(self, args) -> (object, []):
        raise NotImplementedError()

    def n_intermediates(self):
        return len(self.activation_names())

from enum import Enum



class TransformationStrategy(Enum):
    Normal = 1
    Inverse =2
    Both = 3

from .transformation_strategies import PytorchTransformationStrategy



class PytorchActivationsIteratorBase(ActivationsIterator):

    def __init__(self, model: ObservableLayersModule, dataset, transformations: TransformationSet, batch_size=32,
                 num_workers=0, adapter: TransformationAdapter = None, use_cuda=torch.cuda.is_available()):
        '''
        models: a pytorch models that implements the forward_intermediate() method
        dataset: a dataset that yields x,y tuples
        transformations: a list of functions that take a numpy
                        sample and return a transformed one
        '''
        self.strategy=


    def get_transformations(self):
        return self.transformations

    def layer_names(self):
        return self.model.activation_names()

    '''
    Returns the activations of the models by iterating first over transformations and 
    then, for each transformation, over samples
    '''

    def transformations_first(self):
        for t_i, transformation in enumerate(self.transformations):
            dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, drop_last=True, pin_memory=True)
            yield transformation, self.samples_activation(t_i, transformation, dataloader)

    def samples_activation(self, t_i, transformation, dataloader):
        for batch, _ in dataloader:
            if self.use_cuda:
                batch = batch.cuda()
            batch = self.transform_batch(transformation, batch)
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(batch)
                batch_activations = [a.cpu().numpy() for a in batch_activations]
                yield batch, batch_activations

    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def samples_first(self):
        dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, drop_last=True)

        with torch.no_grad():
            for batch, y_true in dataloader:
                batch_cpu = batch
                if self.use_cuda:
                    batch = batch.cuda()
                for i in range(batch.shape[0]):
                    x = batch[i, :]
                    yield batch_cpu[i, :], self.transformations_activations(x)

    def transformations_activations(self, x):

        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        for batch in dataloader:
            y, batch_activations = self.model.forward_intermediates(batch)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch, batch_activations



    def set_strategy(self,ts:TransformationStrategy):
        self.transformation_strategy = ts.value