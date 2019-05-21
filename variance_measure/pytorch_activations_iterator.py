from .activations_iterator import ActivationsIterator
import torch
from torchvision import transforms
from torchvision.transforms import functional
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
#from pytorch.classification_dataset import ImageDataset
import imgaug.augmenters as aug

class ImageDataset(Dataset):


    def __init__(self, image_dataset,rotation=None,translation=None,scale=None,dataformat="NCHW"):


        self.dataset=image_dataset
        self.dataformat=dataformat
        self.rotation=rotation
        self.translation=translation
        self.scale=scale

        self.transform=self.setup_transformation_pipeline(image_dataset,dataformat,rotation,translation,scale)


        # def debug(img):
        #     print(img)
        #     return img
        # transformations.append(transforms.Lambda(debug))


    def setup_transformation_pipeline(self,image_dataset,dataformat,rotation,translation,scale):
        x, y = image_dataset.get_all()

        if dataformat=="NCHW":
            n, c, w, h = x.shape
        elif dataformat=="NHWC":
            n, w, h, c = x.shape
        else:
            raise ValueError

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
                                     #,fillcolor=0)
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

    def update_transformation(self,rotation,translation,scale):
        if not rotation is None:
            self.update_rotation_angle(rotation)
        # TODO add other transformations

    def update_rotation_angle(self,degrees_range):
        self.rotation_transformation.degrees=degrees_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        x,y=self.dataset.get_batch(idx)
        #print("initial batch shape/type:",x.shape,x.dtype)
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
            #print(d.shape)
            images.append(d)

        x=torch.stack(images,dim=0)
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