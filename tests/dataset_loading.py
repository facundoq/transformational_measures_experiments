import tests
print(dir(tests))
print(tests.dataset_loading)

import datasets





dataset="mnist"
#(x_train, y_train), (x_test, y_test), input_shape,num_classes,labels= datasets.get(dataset)
dataformat="NHWC"
d=datasets.get(dataset,dataformat=dataformat)

print(d.summary())




#plot_image_grid(d.x_train,d.y_train,samples=32)
