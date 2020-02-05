
from typing import Tuple
import torch
from transformation_measure import TransformationSet
import transformation_measure as tm

def list_get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]

class InverseTransformer:
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

    def get_inverse_transformations(self, shapes: [Tuple[int,]],
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

    def filter_activations(self,activations: [torch.Tensor])->[torch.Tensor]:
        return list_get_all(activations,self.indices)

    def inverse_trasform_st_same_row(self, activations: [torch.Tensor],
                                     t_start: int, t_end: int):
        # iterate over each layer and corresponding layer transformations
        for layer_activations, layer_transformations in zip(activations, self.inverse_transformation_sets):
            # each sample of the layer activations corresponds to a different column of the st matrix
            # => a different transformation
            # t_start and t_end indicate the corresponding column indices
            for i, inverse in enumerate(layer_transformations[t_start:t_end]):
                inverse_activations = inverse(layer_activations[i,:])
                # print(fm.shape, inverse_fm.shape)
                layer_activations[i,:] = inverse_activations

    def inverse_trasform_st_same_column(self,activations: [torch.Tensor],t_i: int):
        for layer_activations, layer_transformations in zip(activations, self.inverse_transformation_sets):
            # each sample of the layer activations corresponds to a different row of the st matrix
            # => a different sample
            # t_i indicate the corresponding column index, that is, the transformation index
            inverse = layer_transformations[t_i]
            layer_activations[:] = inverse(layer_activations)

