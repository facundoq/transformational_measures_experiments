from testing.utils import plot_image_grid
import datasets

dataset="mnist"
dataformat="NHWC"
d=datasets.get(dataset,dataformat=dataformat)

print(d.summary())

plot_image_grid(d.x_train,d.y_train,samples=32)
