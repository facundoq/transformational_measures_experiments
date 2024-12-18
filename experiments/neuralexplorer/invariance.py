
from genericpath import exists

from numpy import str_
from experiments.neuralexplorer.base import NeuralExplorerExperiment
import torch 
from pathlib import Path
from torchvision import models
import tinyimagenet
import torchvision

import math
import torchvision.transforms.functional
import tmeasures as tm

from testing.util import plot_image_grid
from . import transforms
class TinyImageNetNoLabels(tinyimagenet.TinyImageNet):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt

class Invariance(NeuralExplorerExperiment):

    def get_tinyimagenet(self):
        transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ])


        dataset_nolabels = TinyImageNetNoLabels(root="~/.datasets/tinyimagenet/",split="test", transform=transforms)
        # dataset = tinyimagenet.TinyImageNet(root="~/.datasets/tinyimagenet/",split="test", transform=transforms)

        # Subsample 
        N = 1000
       
        indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=N, stratify=dataset_nolabels.targets,random_state=0)
        dataset_nolabels = Subset(dataset_nolabels, indices)
        return dataset_nolabels
    
    def plot_images(self,images,path):
        grid = torchvision.utils.make_grid(images)

        
        grid_np = grid.permute(1,2,0).numpy()
        plt.figure(dpi=200)
        plt.imshow(grid_np)
        plt.tight_layout()
        plt.savefig(path)
    def plot_transformed_samples(self,dataset:torchvision.datasets.VisionDataset,transformation_name:str,transformation:list[callable]):
        n=500
        step = 50
        transformed_images = [ t(dataset[i]) for i in range(0,n,step) for t in transformation]
        transformation_folderpath = self.folderpath/transformation_name
        transformation_folderpath.mkdir(exist_ok=True,parents=True)
        self.plot_images(transformed_images,transformation_folderpath/"sample.jpg")
        
    def get_activation_module(self,model,device):
        normalized_model = torch.nn.Sequential(
            torchvision.transforms.Normalize(mean=TinyImageNetNoLabels.mean,std=TinyImageNetNoLabels.std),
            model,
        )
        # Put the model in evaluation mode
        normalized_model.eval()
        normalized_model.to(device)

        # Create an ActivationsModule from the vanilla model
        activations_module = tm.pytorch.AutoActivationsModule(normalized_model)
        print(activations_module.activation_names())
        return activations_module

    def compute_measure(activations_module:tm.pytorch,transformation:list[callable],dataset:torchvision.datasets.VisionDataset,device:torch.DeviceObjType,filepath:Path):
        import pickle
        from pathlib import Path
        # Define options for computing the measure
        
        if filepath.exists():
            with open(filepath,"rb") as f:
                measure_result = pickle.load(f)
                print(f"loaded measure results from {filepath}", measure_result) 
        else:
            options = tm.pytorch.PyTorchMeasureOptions(batch_size=128, num_workers=0,model_device=device,measure_device=device,data_device="cpu")

            # Define the measure and evaluate it
            measure = tm.pytorch.NormalizedVarianceInvariance()
            measure_result_pt:tm.pytorch.PyTorchMeasureResult = measure.eval(dataset,transformation,activations_module,options)
            measure_result = measure_result_pt.numpy()
            with open(filepath,"wb") as f:
                pickle.dump(measure_result,f)
    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        dataset = self.get_tinyimagenet()
        transformation_name ="rotation"
        transformation = transforms.transformations[transformation_name]
        self.plot_transformed_samples(dataset,transformation_name,transformation)
        activations_module = self.get_activation_module(model,device)
        filepath = Path(f"{model.__class__.__name__}_{transformation_name}_invariance.pkl")
        self.compute_measure(transformation,dataset,device,filepath)
        

    

    

