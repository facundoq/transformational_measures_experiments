import skimage.io

import datasets
d= datasets.get("cifar10")
imgs = d.x_train.transpose((0,2,3,1))
print(imgs.shape)
for i in range(10):
    skimage.io.imsave(f"testing/cifar{i}.png",imgs[i,:] )