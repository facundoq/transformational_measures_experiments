import datasets
from utils import profiler
from pytorch.numpy_dataset import NumpyDataset
from testing import util
import itertools
import transformation_measure as tm
import matplotlib
from pytorch.pytorch_image_dataset import ImageDataset
matplotlib.use('Agg')
import numpy as np
import config
if __name__ == '__main__':

    model_configs=[config.SimpleConvConfig()]
    dataset_names=["mnist","cifar10","lsa16","rwth"]
    transformations=[tm.SimpleAffineTransformationGenerator(r=360,n_rotations=8)]

    use_cuda=True

    iterator_classes = [
        tm.InvertedPytorchActivationsIterator,
        tm.NormalPytorchActivationsIterator,
        tm.BothPytorchActivationsIterator,
    ]
    batch_size=8
    for dataset_name,model_config,transformation,iterator_class in itertools.product(dataset_names,model_configs,transformations,iterator_classes):
        print(f"### Loading dataset {dataset_name} and model {model_config.name}....")
        print(f"### Transformation {transformation}, iterator {iterator_class.__name__}")
        dataset = datasets.get(dataset_name)
        numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
        image_dataset=ImageDataset(numpy_dataset)

        model = model_config.make_model(dataset.input_shape, dataset.num_classes, use_cuda)

        p= profiler.Profiler()
        p.event("start")

        #transformations=tm.SimpleAffineTransformationGenerator(r=360, s=4, t=3)

        transformation.set_input_shape(dataset.input_shape)
        transformation.set_pytorch(True)
        transformation.set_cuda(use_cuda)

        iterator = iterator_class(model, image_dataset, transformation, batch_size=batch_size, num_workers=0, adapter=None,use_cuda=use_cuda)

        adapter = tm.PytorchNumpyImageTransformationAdapter(use_cuda=use_cuda)
        folderpath = config.testing_path() / f"{iterator.__class__.__name__}"
        folderpath.mkdir(exist_ok=True,parents=True)


        i=0
        for original_x,activations_iterator in iterator.samples_first():

            for v in activations_iterator:
                if iterator_class == tm.BothPytorchActivationsIterator:
                    x_transformed, pre_transformed_activations, post_transformed_activations = v
                    layers=np.concatenate([pre_transformed_activations[0], post_transformed_activations[0]], axis=0)
                else:
                    x_transformed, activations = v
                    layers=activations[0]
                layers= layers.transpose((0, 2, 3, 1))
                # grab only the first feature map
                layers= layers[:, :, :, 0:1]

                filepath=folderpath/ f"{dataset.name}_samples_first_layer_{i}.png"
                util.plot_image_grid(layers, layers.shape[0], show=False, save=filepath)
                i = i + 1
                if i ==10:
                    break
            break


        for transformation,activations_iterator in iterator.transformations_first():

            for v in activations_iterator:
                if iterator_class == tm.BothPytorchActivationsIterator:
                    x_transformed, pre_transformed_activations, post_transformed_activations = v
                    activations = [np.concatenate([a, b], axis=0) for a, b in zip(pre_transformed_activations, post_transformed_activations)]
                else:
                    x_transformed, activations = v

                for i,x in enumerate(activations):
                    if len(x.shape)==4:
                        x= x.transpose((0, 2, 3, 1))
                        # grab only the first feature map
                        x= x[:, :, :, 0:1]
                        filepath=folderpath/ f"{dataset.name}_transformations_first_layer_{i}.png"
                        util.plot_image_grid(x, x.shape[0], show=False, save=filepath)

                break
            break


        p.event("end")
        print(p.summary(human=True))

