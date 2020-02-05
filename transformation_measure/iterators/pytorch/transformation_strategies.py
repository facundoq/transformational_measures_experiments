import abc
from torch.utils.data import DataLoader
from transformation_measure.iterators.pytorch_image_dataset import ImageDataset

from transformation_measure import TransformationSet,Transformation
import torch
from transformation_measure import TransformationSet,Transformation
from transformation_measure.adapters import TransformationAdapter

class PytorchTransformationStrategy(abc.ABC):

    def __init__(self, model: ObservableLayersModule, dataset, transformations: TransformationSet, batch_size=32,
                 num_workers=0, adapter: TransformationAdapter = None, use_cuda=torch.cuda.is_available()):

        self.model = model
        self.dataset = dataset
        self.transformations = transformations

        self.image_dataset = ImageDataset(self.dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.adapter = adapter

    @abc.abstractmethod
    def samples_activation(self, t_i:int, transformation:Transformation, dataloader:DataLoader):
        pass

    @abc.abstractmethod
    def transformations_activations(self, x:torch.Tensor):
        pass

    def transform_sample(self, x: torch.Tensor):

        x = x.unsqueeze(0)
        results = []
        for i, transformation in enumerate(self.transformations):
            transformed = self.transform_batch(transformation, x)
            results.append(transformed)
        return torch.cat(results)

    def transform_batch(self, transformation, x: torch.Tensor):
        if not self.adapter is None:
            x = self.adapter.pre_adapt(x)
        x = transformation(x)
        if not self.adapter is None:
            x = self.adapter.post_adapt(x)
        return x



class NormalStrategy(PytorchTransformationStrategy):

    def samples_activation(self,t_i,transformation,dataloader):
        for batch, _ in dataloader:
            if self.use_cuda:
                batch=batch.cuda()
            batch=self.transform_batch(transformation,batch)
            with torch.no_grad():
                y, batch_activations = self.model.forward_intermediates(batch)
                batch_activations =  [a.cpu().numpy() for a in batch_activations]
                yield batch,batch_activations

    def transformations_activations(self,x):

        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        for batch in dataloader:
            y, batch_activations = self.model.forward_intermediates(batch)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch,batch_activations





