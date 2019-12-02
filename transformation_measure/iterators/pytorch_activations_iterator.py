from .activations_iterator import ActivationsIterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from .pytorch_image_dataset import ImageDataset
import numpy as np

from transformation_measure import TransformationSet

class PytorchActivationsIterator(ActivationsIterator):

    def __init__(self, model:nn.Module, dataset, transformations:TransformationSet, batch_size=32,num_workers=0):
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
        self.use_cuda=torch.cuda.is_available()

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
            dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True,pin_memory=True)
            yield transformation, self.samples_activation(transformation,dataloader)


    def samples_activation(self,transformation,dataloader):
        for x, y_true in dataloader:
            x=self.transform_batch(transformation,x)
            x_transformed = x.numpy()
            if self.use_cuda:
                x = x.cuda()
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(x)
                batch_activations =  [a.cpu().numpy() for a in batch_activations]
                yield x_transformed,batch_activations

    def transform_batch(self,transformation,x:torch.Tensor):
        # to NHWC order
        #x=x.permute(0,2,3,1)
        for i in range(x.shape[0]):
            sample_np=x[i,:].numpy()
            transformed_np = transformation(sample_np)
            x[i,:] = torch.from_numpy(transformed_np)
        # To NCHW order
        #x = x.permute(0,3,1,2)
        return x

    def transform_sample(self,x:torch.Tensor):
        all = torch.empty((len(self.transformations), *x.shape))
        sample_np = x.numpy()
        for i,transformation in enumerate(self.transformations):
            transformed_np=transformation(sample_np)
            all[i,:]=torch.from_numpy(transformed_np)
        return all

    def transformations_activations(self,x):
        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, drop_last=False)
        i_start=0
        for batch in dataloader:
            i_end = i_start + batch.shape[0]
            batch_numpy = x_transformed[i_start:i_end, :].numpy()
            i_start=i_end
            if self.use_cuda:
                batch=batch.cuda()
            y, batch_activations = self.model.forward_intermediates(batch)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch_numpy,batch_activations

        #     for i,a in enumerate(batch_activations.detach().cpu().numpy()):
        #         # print(a.detach().cpu().numpy().shape)
        #         # print(self.activation_names()[i],"=> ",a.detach().cpu().numpy().shape)
        #         activations[i].append(a.)
        # activations=[ np.vstack(a) for a in activations]
        # n = x.shape[0]
        # if self.use_cuda:
        #     x= x.cuda()
        # else:
        #     x=x_transformed
        # b=self.batch_size
        # if n>b:
        #     b=n
        # batches=n//b
        # for i in range(batches):
        #     y, batch_activations = self.model.forward_intermediates(batch)
        #     batch=x[i*b:(i+1)*b,]
        #     batch_activations = [a.detach().cpu().numpy() for a in batch_activations]
        #     activations.append(batch_activations)
        #
        # remaining = n - batches * self.batch_size
        # if remaining>0:
        #     batch = x[-remaining:, ]
        #     y, batch_activations = self.model.forward_intermediates(batch)
        #     batch_activations = [a.detach().cpu().numpy() for a in batch_activations]
        #     activations.append(batch_activations)
        #
        # return activations,x_transformed

    '''
        Returns the activations of the models by iterating first over transformations and 
        then, for each transformation, over samples
    '''
    def samples_first(self):
        dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)

        with torch.no_grad():
            for x, y_true in dataloader:
                for i in range(x.shape[0]):
                    sample=x[i, :]
                    yield sample,self.transformations_activations(sample)



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
