import torch
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import transformation_measure as tm

class ImageDataset(Dataset):

    def __init__(self, image_dataset, transformations:tm.TransformationSet=tm.SimpleAffineTransformationGenerator(), dataformat="NCHW"):

        self.dataset=image_dataset
        self.dataformat=dataformat
        self.transformations=list(transformations)

        self.setup_transformation_pipeline()

    def setup_transformation_pipeline(self,):
        x, y = self.dataset.get_all()

        if self.dataformat=="NCHW":
            n, c, w, h = x.shape

        elif self.dataformat=="NHWC":
            n, w, h, c = x.shape
        else:
            raise ValueError(f"Unsupported data format: {self.dataformat}.")
        self.h = h
        self.w = w
        self.mu, self.std = self.calculate_mu_std(x, self.dataformat)

        # transformations = [transforms.ToPILImage(), ]
        #
        # transformations.append(transforms.ToTensor())
        # transformations.append(transforms.Normalize(mu, std))
        # return transforms.Compose(transformations)

    def calculate_mu_std(self,x,dataformat):

        # mu = image_dataset.mean()/255
        # std = image_dataset.std()/255

        xf = x.float()
        if dataformat == "NCHW":
            dims = (0, 2, 3)
        elif dataformat == "NHWC":
            dims = (0, 1, 2)
        else:
            raise ValueError()

        mu = xf.mean(dim=dims) / 255
        std = xf.std(dim=dims) / 255

        std[std == 0] = 1
        return mu.numpy(), std.numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        x,y=self.get_batch(idx)
        # print(y.shape)
        return x[0,],y

    def transform_nchw(self,x):
        if self.dataformat == "NCHW":
            pass
        elif self.dataformat == "NHWC":
            x = x.permute([0, 3, 1, 2])
        else:
            raise ValueError()
        return x

    def transform_nhwc(self,x):
        if self.dataformat == "NCHW":
            pass
        elif self.dataformat == "NHWC":
            x = x.permute(0,2,3,1)
        else:
            raise ValueError()
        return x

    def transform_batch(self,x):
        nt = len(self.transformations)
        # to NHWC order
        x = self.transform_nhwc(x).float()
        for i in range(x.shape[0]):
            sample_np=x[i,:].numpy()
            sample_np = (sample_np - self.mu) / self.std
            t = self.transformations[np.random.randint(0, nt)]
            transformed_np = t(sample_np)
            x[i,:] = torch.from_numpy(transformed_np)
        # To NCHW order
        x = self.transform_nchw(x)
        return x

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        x,y=self.dataset.get_batch(idx)
        x=self.transform_batch(x)
        y=y.type(dtype=torch.LongTensor)
        return x,y


    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)
