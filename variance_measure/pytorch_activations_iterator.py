from .activations_iterator import ActivationsIterator
import torch
from torchvision import transforms
import torch

from torch.utils.data import Dataset,DataLoader
#from pytorch.classification_dataset import ImageDataset


class ImageDataset(Dataset):


    def __init__(self, image_dataset,rotation=None,translation=None,scale=None):


        self.dataset=image_dataset
        x,y=image_dataset.get_all()
        mu = x.mean(axis=(0, 1, 2))/255
        std = x.std(axis=(0, 1, 2))/255
        std[std == 0] = 1

        transformations=[transforms.ToPILImage(),
                         transforms.ToTensor(),
                         transforms.Normalize(mu, std),
                        ]

        if not rotation is None:
            rotation_transformation=transforms.RandomRotation(rotation, resample=Image.BILINEAR)
            transformations.insert(1,rotation_transformation)

        if not translation is None:
            translation_transformation=transforms.CenterCrop(translation)
            transformations.insert(1,translation_transformation)

        if not scale is None:
            scale_transformation=transforms.Scale(scale)
            transformations.insert(1,scale_transformation)

        self.transform=transforms.Compose(transformations)


    def update_transformation(self,rotation,translation,scale):
        if not rotation is None:
            self.update_rotation_angle(rotation)
        # TODO add other transformations

    def update_rotation_angle(self,degrees_range):
        self.rotation_transformation.degrees=degrees_range

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        images = []
        x,y=self.dataset.get_batch(idx)
        x=x.copy()
        y=y.copy()
        for i in idx:
            self.x[i, :]= self.transform(self.x[i, :])
        #y = torch.from_numpy(self.y[idx, :].argmax(axis=1))
        return x,y

    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)



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

        return zip(rotation,translation,scale)

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