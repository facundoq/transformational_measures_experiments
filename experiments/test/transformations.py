import transformational_measures.measure
from ..invariance.common import *
from .base import TestExperiment

import datasets
import torch
from utils import profiler
from pytorch.numpy_dataset import NumpyDataset
import config
import transformational_measures as tm
import matplotlib
from testing import util

class PytorchTransformations(TestExperiment):

    def description(self):
        return """Test pytorch transformations and generate sample images"""

    def run(self):
        dataset_names = ["mnist","cifar10"]
        cudas = [False, True]
        r, s, t, combined = config.common_transformations_combined
        transformation_sets = [r,s,t, combined]
        n_images = 4
        p = profiler.Profiler()
        p.event("start")

        for transformations,dataset_name,use_cuda in itertools.product(transformation_sets,dataset_names,cudas):
            print(f"### Loading dataset {dataset_name} ....")
            folderpath = self.folderpath / f"{dataset_name}"
            folderpath.mkdir(exist_ok=True, parents=True)
            dataset = datasets.get_classification(dataset_name)
            dataset.normalize_features()
            adapter = tm.NumpyPytorchImageTransformationAdapter(use_cuda=use_cuda)
            numpy_dataset = NumpyDataset(dataset.x_test, dataset.y_test)
            x, y = numpy_dataset.get_batch(range(n_images))
            if use_cuda:
                x = x.cuda()

            n_t = len(transformations)
            print(f"Dataset {dataset_name}, Transformations: {n_t} ({transformations})")
            for i in range(n_images):
                print(f"Generating plots for image {i}")
                original_torch = x[i, :]
                # transformed_images = []
                transformed_torch = torch.zeros((n_t, *original_torch.shape))
                original_torch = original_torch.unsqueeze(0)
                for j, t in enumerate(transformations):
                    transformed_torch[j, :] = t(original_torch)
                transformed_numpy = adapter.post_adapt(transformed_torch)
                cuda_str = "_cuda" if use_cuda else ""
                filepath = folderpath / f"samples_first_{i}_{transformations}{cuda_str}.png"
                util.plot_image_grid(transformed_numpy, samples=n_t, grid_cols=16, show=False, save=filepath)

        p.event("end")
        print(p.summary(human=True))






