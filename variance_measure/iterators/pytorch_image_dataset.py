import torch
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from PIL import Image



class ImageDataset(Dataset):

    def __init__(self, image_dataset,rotation=None,translation=None,scale=None,dataformat="NCHW"):

        self.dataset=image_dataset
        self.dataformat=dataformat

        self.rotation=rotation
        self.translation=translation
        self.scale=scale

        self.transform=self.setup_transformation_pipeline(image_dataset,dataformat,rotation,translation,scale)

    def setup_transformation_pipeline(self,image_dataset,dataformat,rotation,translation,scale):
        x, y = image_dataset.get_all()

        if dataformat=="NCHW":
            n, c, w, h = x.shape

        elif dataformat=="NHWC":
            n, w, h, c = x.shape
        else:
            raise ValueError
        self.h = h
        self.w = w
        mu, std = self.calculate_mu_std(x, dataformat)

        transformations = [transforms.ToPILImage(), ]

        if rotation is None:
            rotation = 0
        if translation is None:
            translation = (0,0)
        if scale is None:
            scale = 1



        def affine_transform(image):
            return functional.affine(image,shear=0,angle=rotation,translate=translation,
                              scale=scale,resample=Image.BILINEAR)
        affine=transforms.Lambda(affine_transform)
        transformations.append(affine)

        if not translation is None or not scale is None:
            scale_transformation = transforms.Resize((h,w))
            transformations.append(scale_transformation)

        transformations.append(transforms.ToTensor())
        transformations.append(transforms.Normalize(mu, std))
        return transforms.Compose(transformations)

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
        return mu,std
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        x,y=self.get_batch(idx)
        # print(y.shape)
        return x[0,],y

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        x,y=self.dataset.get_batch(idx)
        images=[]

        # put image in NCHW format for PIL
        if self.dataformat=="NCHW":
            pass
        elif self.dataformat=="NHWC":
            x=x.permute([0, 3, 1, 2])
        else:
            raise ValueError()
        for i in range(x.shape[0]):
            d=self.transform(x[i,:,:,:])
            images.append(d)

        x=torch.stack(images,dim=0)
        y=y.type(dtype=torch.LongTensor)
        # print(x.shape,y.shape)
        return x,y


    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)
