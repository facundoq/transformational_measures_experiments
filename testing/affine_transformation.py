import skimage.io
import transformational_measures as tm
import numpy as np
import matplotlib.pyplot as plt
import os
import config


results_path= config.testing_path() / "affine_transformation/"
results_path.mkdir(parents=True,exist_ok=True)

use_cuda=True
source_path="testing/samples/mnist.png"

input_shape=np.array((28,28,1))

def apply_transformation_numpy(t, image_name):
    x= skimage.io.imread(source_path)
    x = x[np.newaxis,:, :, np.newaxis]
    x= t(x)
    filepath = results_path / f"{image_name}_numpy.png"
    skimage.io.imsave( filepath, x)

def apply_transformation_pytorch(t, image_name):

    adapter = tm.NumpyPytorchImageTransformationAdapter(use_cuda=use_cuda)
    x = skimage.io.imread(source_path) / 255.0
    x = x[np.newaxis, :, :, np.newaxis].astype(np.float32)
    x = adapter.pre_adapt(x)
    x = t(x)
    x = adapter.post_adapt(x)
    x = x[0, :, :, 0]
    x = (x * 255).astype(np.uint8)
    filepath = results_path / f"{image_name}_pytorch.png"
    skimage.io.imsave(filepath, x)

def apply_transformation(p,image_name):
    np_t= tm.AffineTransformationNumpy(p, input_shape)
    apply_transformation_numpy(np_t,image_name)
    apply_transformation_numpy(np_t.inverse(), f"{image_name}_inverse")

    pt_t = tm.AffineTransformationPytorch(p, input_shape, use_cuda=use_cuda)
    apply_transformation_pytorch(pt_t, image_name)
    apply_transformation_pytorch(pt_t.inverse(), f"{image_name}_inverse")


p=(0, (0, 0) , (1, 1))
apply_transformation(p, "identity")

for r in [0,45,90,135,180,360]:
    p=(r/360*2*3.14, (0, 0), (1, 1))
    apply_transformation(p, f"rotation{r}")

for s in [0.1,0.5,0.8]:
    for s2 in [0.1, 0.5, 0.8]:
        p=(0, (0, 0), (s,s2))
        apply_transformation(p, f"resize={s}-{s2}")

p=(0, (5, 5), (1, 1))
apply_transformation(p, "translation55px")

p=(0, (0, 5), (1, 1))
apply_transformation(p, "translation50px")

p=(0, (5, 0), (1, 1))
apply_transformation(p, "translation05px")
