import datasets
import torch
from utils import profiler
from pytorch.numpy_dataset import NumpyDataset
import config
import transformation_measure as tm
import matplotlib

matplotlib.use('Agg')


dataset_name="mnist"

print(f"### Loading dataset {dataset_name} ....")

use_cuda=False
dataset = datasets.get(dataset_name)
dataset.normalize_features()
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
p= profiler.Profiler()
p.event("start")

r,s,t,combined=config.common_transformations_combined
transformation_sets = [t,combined]
# transformations.set_input_shape(dataset.input_shape)
# transformations.set_cuda(use_cuda)
# transformations.set_pytorch(True)

adapter = tm.NumpyPytorchImageTransformationAdapter(use_cuda=use_cuda)
folderpath = config.testing_path() / f"affine_transformation_generator_pytorch/{dataset_name}"
folderpath.mkdir(exist_ok=True,parents=True)

n_images=4
i=0
x,y= numpy_dataset.get_batch(range(n_images))
if use_cuda:
    x=x.cuda()
from testing import util

for transformations in transformation_sets:
    n_t=len(transformations)
    print(f"Transformations: {n_t} ({transformations})")
    for i in range(n_images):
        print(f"Generating plots for image {i}")
        original_torch = x[i, :]
        #transformed_images = []
        transformed_torch= torch.zeros((n_t, *original_torch.shape))
        original_torch= original_torch.unsqueeze(0)
        for j,t in enumerate(transformations):
            transformed_torch[j, :]=t(original_torch)

        transformed_numpy = adapter.post_adapt(transformed_torch)
        cuda_str = "_cuda" if use_cuda else ""
        filepath = folderpath / f"samples_first_{i}_{transformations}{cuda_str}.png"
        util.plot_image_grid(transformed_numpy, samples = n_t, grid_cols=16, show=False, save=filepath)

p.event("end")
print(p.summary(human=True))

