from .activations_iterator import ActivationsIterator

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from transformation_measure import TransformationSet
from transformation_measure.adapters import TransformationAdapter
import transformation_measure as tm
import numpy as np

def list_get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]

class ActivationsInverter:
    def __init__(self,activation_shapes: [Tuple[int,]], layer_names: [str],
                         transformation_set: tm.TransformationSet):

        self.indices,self.layer_names=self.get_valid_layers(activation_shapes,layer_names,transformation_set)
        self.valid_shapes=list_get_all(activation_shapes,self.indices)
        self.inverse_transformation_sets=self.get_inverse_transformations(self.valid_shapes,transformation_set)


    def get_valid_layers(self, activation_shapes: [Tuple[int,]], layer_names: [str],
                         transformation_set: tm.TransformationSet):
        # get indices of layers for which the transformation is valid
        indices = [i for i, shape in enumerate(activation_shapes) if transformation_set.valid_input(shape)]
        # keep only this layers
        layer_names = list_get_all(layer_names, indices)
        return layer_names, indices

    def get_inverse_transformations(self, shapes: [np.ndarray],
                                    transformation_set: tm.TransformationSet):
        inverse_transformation_sets = []

        for s in shapes:
            n, c, h, w = s
            layer_transformation_set: tm.TransformationSet = transformation_set.copy()
            # layer_transformation_set.set_pytorch(False)
            layer_transformation_set.set_input_shape((h, w, c))
            layer_transformation_set_list = [l.inverse() for l in layer_transformation_set]
            inverse_transformation_sets.append(layer_transformation_set_list)
        return inverse_transformation_sets

    def inverse_trasform_feature_maps(self, activations: [torch.Tensor],
                                      t_start: int, t_end: int) -> [torch.Tensor]:
        for layer, layer_transformations in zip(activations, self.inverse_transformation_sets):
            for i, inverse in enumerate(layer_transformations[t_start:t_end]):
                # ax[0, i].imshow(layer[i,0,:,:],cmap="gray")
                # ax[0, i].axis("off")
                # print(inverse.__class__,layer.__class__)
                #fm = layer[i:i + 1, :].transpose(0, 2, 3, 1)
                activation=layer[i:i + 1, :]
                inverse_fm = inverse(activation)
                #inverse_fm = inverse_fm.transpose(0, 3, 1, 2)
                # print(fm.shape, inverse_fm.shape)
                layer[i, :] = inverse_fm[0, :]




class PytorchActivationsIteratorInverted(tm.iterators.pytorch_activations_iterator.PytorchActivationsIterator):

    def get_transformations(self):
        return self.transformations
    def layer_names(self):
        return self.model.activation_names()

    # '''
    # Returns the activations of the models by iterating first over transformations and
    # then, for each transformation, over samples
    # '''
    # def transformations_first(self):
    #     for transformation in self.transformations:
    #         dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True,pin_memory=True)
    #         yield transformation, self.samples_activation(transformation,dataloader)

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

    # def samples_first(self):
    #     dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,drop_last=True)
    #
    #     with torch.no_grad():
    #         for batch, y_true in dataloader:
    #             batch_cpu = batch
    #             if self.use_cuda:
    #                 batch = batch.cuda()
    #             for i in range(batch.shape[0]):
    #                 x = batch[i, :]
    #                 yield batch_cpu[i,:], self.transformations_activations(x)

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


    def get_inverted_activations_iterator(self) -> ActivationsIterator:
        return self



