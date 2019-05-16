from .activations_iterator import ActivationsIterator
import torch

from torch.utils.data import Dataset,DataLoader
#from pytorch.classification_dataset import ImageDataset

class PytorchActivationsIterator(ActivationsIterator):

    def __init__(self, model, dataset, transformations, config):
        super().__init__(model, dataset, transformations)
        self.model_config=config

    def activation_count(self):
        n_intermediates = self.model.n_intermediates()

    def generate_transformations(self):

        # logging.debug(f"    Rotation {degrees}...")
        scale=[None for r in self.transformations["rotation"]]
        translation=[None for r in self.transformations["rotation"]]
        rotation=[(r - 1, r + 1) for r in self.transformations["rotation"]]

        return rotation,translation,scale

    def transformations_first(self,batch_size):
        dataloader= DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        transformation_list=self.generate_transformations()
        for transformation in transformation_list:
            rotation,translation,scale=transformation
            self.dataset.update_transformation(rotation,translation,scale)
            print(rotation)
            yield transformation,1
            # for x, y_true in dataloader:
            #     if self.model_config.use_cuda:
            #         x = x.cuda()
            #     with torch.no_grad():
            #         y, batch_activations = self.model.forward_intermediates(x)
            #         batch_activations = batch_activations.detach().cpu().numpy()
            #         yield batch_activations
            #


    def samples_first(self,batch_size):
        pass