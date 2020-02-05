from transformation_measure.iterators.activations_iterator import ActivationsIterator

import torch
from torch.utils.data import DataLoader

import transformation_measure as tm
from .inverse_transformer import InverseTransformer

class PytorchActivationsIteratorBoth(tm.iterators.pytorch.activations_iterator_base.PytorchActivationsIteratorBase):

    def samples_activation(self,t_i,transformation,dataloader):
        for batch, _ in dataloader:
            if self.use_cuda:
                batch=batch.cuda()
            batch=self.transform_batch(transformation,batch)
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(batch)
                if self.inverse_transformer is None:
                    shapes = [a.shape for a in batch_activations]
                    self.inverse_transformer = InverseTransformer(shapes, self.layer_names(),
                                                                  self.get_transformations())
                # filter activations to those accepted by the transformations
                batch_activations = self.inverse_transformer.filter_activations(batch_activations)
                inverted_batch_activations=batch_activations.copy()
                # inverse transform selected activations
                self.inverse_transformer.inverse_trasform_st_same_column(inverted_batch_activations,t_i)
                batch_activations =  [a.cpu().numpy() for a in batch_activations]
                inverted_batch_activations = [a.cpu().numpy() for a in inverted_batch_activations]
                yield batch,batch_activations,inverted_batch_activations
    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def transformations_activations(self,x):
        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        self.inverse_transformer=None
        t_start=0
        for batch in dataloader:
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(batch)
            t_end=t_start+batch_activations.shape[0]
            if self.inverse_transformer is None:
                shapes = [a.shape for a in batch_activations ]
                self.inverse_transformer=InverseTransformer(shapes, self.layer_names(), self.get_transformations())
            # filter activations to those accepted by the transformations
            batch_activations=self.inverse_transformer.filter_activations(batch_activations)
            inverted_batch_activations = batch_activations.copy()
            # inverse transform selected activations
            self.inverse_transformer.inverse_trasform_st_same_row(inverted_batch_activations, t_start, t_end)
            t_start = t_end
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            inverted_batch_activations = [a.cpu().numpy() for a in inverted_batch_activations]
            yield batch, batch_activations, inverted_batch_activations

    def get_inverted_activations_iterator(self) -> ActivationsIterator:
        return self

    def get_inverted_and_normal_iterator(self) -> ActivationsIterator:
        return self

    def layer_names(self)->[str]:
        return self.inverse_transformer.layer_names


