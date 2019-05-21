import tests
print(dir(tests))
print(tests.dataset_loading)

import datasets

dataset="mnist"
dataformat="NHWC"
d=datasets.get(dataset,dataformat=dataformat)

print(d.summary())




#plot_image_grid(d.x_train,d.y_train,samples=32)
