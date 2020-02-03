from testing.util import plot_image_grid
import datasets
import numpy as np

dataset="lsa16"
d=datasets.get(dataset)

print(d.summary())

(unique, counts) = np.unique(d.y_test, return_counts=True)

print("Test")
for i,c in zip(unique,counts):
    print(f"Class {i:02}, {c:005} samples ")

(unique, counts) = np.unique(d.y_train, return_counts=True)
print("train")
for i,c in zip(unique,counts):
    print(f"Class {i:02}, {c:005} samples ")


#plot_image_grid(d.x_train,d.y_train,samples=32)
