from .activations_iterator import ActivationsIterator

import torch
from torch.utils.data import DataLoader
from .pytorch_image_dataset import ImageDataset


class PytorchActivationsIterator(ActivationsIterator):

    def __init__(self, model, dataset, transformations, batch_size=32):
        '''
        model: a pytorch model that implements the forward_intermediate() method
        dataset: a dataset that yields x,y tuples
        transformations: a list of functions that take a numpy
                        sample and return a transformed one
        '''
        super().__init__(model, dataset, transformations)
        self.image_dataset=ImageDataset(self.dataset)
        self.batch_size=batch_size

    def activation_names(self):
        return self.model.activation_names()

    def transformations_first(self):
        for transformation in self.transformations:
            dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)
            yield transformation, self.samples_activation(transformation,dataloader)


    def samples_activation(self,transformation,dataloader):
        for x, y_true in dataloader:
            x=self.transform_batch(transformation,x)
            if torch.cuda.is_available():
                x = x.cuda()
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(x)
                batch_activations = [ a.detach().cpu().numpy() for a in batch_activations]
                yield x.cpu().numpy(),batch_activations

    def transform_batch(self,transformation,x):
        # to NHWC order
        x=x.permute(0,2,3,1)
        for i in range(x.shape[0]):
            sample_np=x[i,:].numpy()
            transformed_np = transformation(sample_np)
            x[i,:] = torch.from_numpy(transformed_np)
        # To NCHW order
        x = x.permute(0,3,1,2)
        return x

    def transform_sample(self,x):
        all = torch.empty((len(self.transformations), *x.shape))
        sample_np = x.numpy().transpose((1,2,0))
        for i,transformation in enumerate(self.transformations):
            transformed_np=transformation(sample_np ).transpose((2,0,1))
            all[i,:]=torch.from_numpy(transformed_np)
        return all

    def transformations_activations(self,x):
        x_transformed = self.transform_sample(x)

        if torch.cuda.is_available():
            x = x_transformed.cuda()
        else:
            x = x_transformed
        with torch.no_grad():
            y, batch_activations = self.model.forward_intermediates(x)
            batch_activations = [a.detach().cpu().numpy() for a in batch_activations]
            return batch_activations,x_transformed

    def samples_first(self):
        dataloader = DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)

        for x, y_true in dataloader:
            for i in range(x.shape[0]):
                activations,x_transformed=self.transformations_activations(x[i, :])
                yield activations,x_transformed.numpy()
