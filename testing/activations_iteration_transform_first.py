import datasets
import utils
from pytorch.numpy_dataset import NumpyDataset
from testing import util
import transformation_measure as tm
import matplotlib
import config
from transformation_measure.iterators.pytorch.test import ImageDataset
matplotlib.use('Agg')


dataset_name="cifar10"
model_config=config.SimpleConvConfig()

print(f"### Loading dataset {dataset_name} and model {model_config.name}....")

use_cuda=True
dataset = datasets.get(dataset_name)
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset)
model,optimizer = model_config.make_model_and_optimizer(dataset.input_shape, dataset.num_classes, use_cuda)
p= utils.Profiler()
p.event("start")

#transformations=tm.SimpleAffineTransformationGenerator(r=360, s=3, t=2,n_rotations=4)
transformations=tm.SimpleAffineTransformationGenerator(r=360,n_rotations=4)
transformations.set_input_shape(dataset.input_shape)
transformations.set_pytorch(True)
transformations.set_cuda(use_cuda)

iterator = tm.NormalPytorchActivationsIterator(model, image_dataset, transformations, batch_size=64, num_workers=0, use_cuda=use_cuda)
adapter = tm.PytorchNumpyImageTransformationAdapter(use_cuda=use_cuda)
folderpath = config.testing_path() / "activations_iterator"
folderpath.mkdir(exist_ok=True,parents=True)

p.event("start")
i=0
for transformation,batch_activations in iterator.transformations_first():
    print(transformation)
    for x,batch_activation in batch_activations:
        x = adapter.pre_adapt(x)
        filepath = folderpath / f"transformation_first{i}.png"

        util.plot_image_grid(x, x.shape[0], show=False, save=filepath)
        i=i+1
        break
p.event("end")
print(p.summary(human=True))
