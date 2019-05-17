from pytorch import classification_dataset

dataset = classification_dataset.get_dataset("mnist")
print(dataset)

from pytorch.numpy_dataset import NumpyDataset
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
print(numpy_dataset)

for i in range(3):
    batch= numpy_dataset.get_batch(list(range(i*32,(i+1)*32)))
    print([v.shape for v in list(batch)])




numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
print(numpy_dataset)

for i in range(3):
    batch= numpy_dataset.get_batch(list(range(i*32,(i+1)*32)))
    print([v.shape for v in list(batch)])

