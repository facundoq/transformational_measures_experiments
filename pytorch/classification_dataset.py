from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

class ClassificationDataset:
    def __init__(self,name,x_train,x_test,y_train,y_test,num_classes,input_shape,labels):
        self.name=name
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.num_classes=num_classes
        self.input_shape=input_shape
        self.labels=labels
    def summary(self):
        result=""
        result+=f"x_train: {self.x_train.shape}, {self.x_train.dtype}\n"
        result+=f"x_test: {self.x_test.shape}, {self.x_test.dtype}\n"
        result+=f"y_train: {self.y_train.shape}, {self.y_train.dtype}\n"
        result+=f"y_test: {self.y_test.shape}, {self.y_test.dtype}\n"
        result+=f"Classes {np.unique(self.y_train.argmax(axis=1))}\n"
        result+=f"min class/max class: {self.y_train.min()} {self.y_train.max()}"
        return result

def check_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


class ImageDataset(Dataset):


    def __init__(self, dataset,rotation=None,translation=None,scale=None):


        self.dataset=dataset
        x,y=dataset.get_all()
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
        image =self.x[idx,:,:,:]
        image = self.transform(image)
        target=self.y[idx,:].argmax()
        return (image,target)

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        images = []
        for i in idx:
            image = self.transform(self.x[i, :, :, :])
            images.append(image)
        y = torch.from_numpy(self.y[idx, :].argmax(axis=1))
        x = torch.stack(images,0)
        return x,y
    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)


def get_data_generator(x,y,batch_size):
    image_dataset=ImageDataset(x,y)
    dataset=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    image_rotated_dataset = ImageDataset(x, y, rotation=180)
    rotated_dataset = DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=1)

    return dataset,rotated_dataset

import datasets

def get_dataset(name):
    (x_train, y_train), (x_test, y_test), input_shape, num_classes, labels = datasets.get_data(name)
    return ClassificationDataset(name, x_train, x_test, y_train, y_test, num_classes, input_shape,labels)