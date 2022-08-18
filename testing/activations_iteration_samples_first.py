import datasets
from utils import profiler
from pytorch.numpy_dataset import NumpyDataset
from testing import util

import tmeasures as tm
import matplotlib
from pytorch.pytorch_image_dataset import ImageClassificationDataset
matplotlib.use('Agg')

import config
model_config=config.SimpleConvConfig()
dataset_name="mnist"


print(f"### Loading dataset {dataset_name} and model {model_config.name}....")

use_cuda=True
dataset = datasets.get_classification(dataset_name)
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageClassificationDataset(numpy_dataset)

model,optimizer = model_config.make_model_and_optimizer(dataset.input_shape, dataset.num_classes, use_cuda)

p= profiler.Profiler()
p.event("start")

#transformations=tm.SimpleAffineTransformationGenerator(r=360, s=4, t=3)
transformations=tm.SimpleAffineTransformationGenerator(r=360,n_rotations=4)
transformations.set_input_shape(dataset.input_shape)
transformations.set_pytorch(True)
transformations.set_cuda(use_cuda)

iterator = tm.NormalPytorchActivationsIterator(model, image_dataset, transformations, batch_size=64, num_workers=0, use_cuda=use_cuda)
adapter = tm.PytorchNumpyImageTransformationAdapter(use_cuda=use_cuda)
folderpath = config.testing_path() / f"activations_iterator"
folderpath.mkdir(exist_ok=True,parents=True)

batch_size=64
i=0
for original_x,activations_iterator in iterator.samples_first():

    for x_transformed,activations in activations_iterator:
        x=adapter.pre_adapt(x_transformed)
        filepath=folderpath/ f"{dataset}_samples_first.png"
        util.plot_image_grid(x, x.shape[0], show=False, save=filepath)
        i = i + 1
        if i ==10:
            break
    break
p.event("end")
print(p.summary(human=True))
