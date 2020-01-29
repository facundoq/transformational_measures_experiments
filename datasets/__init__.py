from . import cluttered_mnist,mnist_rot,mnist,fashion_mnist,cifar10
from typing import List

import numpy as np
import os
from enum import Enum

class DatasetSubset(Enum):
    train="train"
    test="test"
    values=[train,test]


class ClassificationDataset:
    def __init__(self, name: str,
                 x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                 num_classes: int, input_shape: np.ndarray, labels: List[str], dataformat: str):
        self.name = name
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.labels = labels
        self.dataformat = dataformat

    def size(self,subset:DatasetSubset):
        if subset==DatasetSubset.test:
            return self.y_test.shape[0]
        elif subset==DatasetSubset.train:
            return self.y_train.shape[0]
        else:
            raise ValueError(subset)

    def reduce_size_subset_stratified(self, percentage, x, y,random=False):
        x1=[]
        y1=[]
        x2=[]
        y2=[]
        for i in range(self.num_classes):
            class_indices = y == i
            class_x=x[class_indices,:]
            class_y=y[class_indices]
            class_n=len(class_y)
            if random:
                indices = np.random.permutation(class_n)
            else:
                indices = list(range(class_n))
            limit=int(class_n * percentage)

            indices1 = indices[:limit]
            x1.append(class_x[indices1,:])
            y1.append(class_y[indices1])

            indices2 = indices[limit:]
            x2.append(class_x[indices2, :])
            y2.append(class_y[indices2])

        x1 = np.vstack(x1)
        y1 = np.hstack(y1)
        x2 = np.vstack(x2)
        y2 = np.hstack(y2)
        return x1, y1, x2, y2

    def reduce_size_stratified(self, percentage):
        if percentage==1:
            return self
        x_train, y_train, _, _ = self.reduce_size_subset_stratified(percentage, self.x_train, self.y_train)
        x_test, y_test, _, _ = self.reduce_size_subset_stratified(percentage, self.x_test, self.y_test)
        return ClassificationDataset(self.name, x_train, x_test, y_train, y_test
                                     , self.num_classes, self.input_shape, self.labels, self.dataformat)

    def summary(self):
        result = f"Image Classification Dataset {self.name}\n" \
            f"Dataformat {self.dataformat}\n"
        result += f"x_train: {self.x_train.shape}, {self.x_train.dtype}\n"
        result += f"x_test: {self.x_test.shape}, {self.x_test.dtype}\n"
        result += f"y_train: {self.y_train.shape}, {self.y_train.dtype}\n"
        result += f"y_test: {self.y_test.shape}, {self.y_test.dtype}\n"
        result += f"Classes {np.unique(self.y_train)}\n"
        result += f"min class/max class: {self.y_train.min()} {self.y_train.max()}"
        return result


datasets={"mnist":mnist
          # ,"fashion_mnist":fashion_mnist
          ,"cifar10":cifar10
          # ,"mnist_rot":mnist_rot
          # ,"cluttered_mnist":cluttered_mnist
          }
names=datasets.keys()


def get(dataset,dataformat="NCHW",path=os.path.expanduser("~/.datasets/")) -> ClassificationDataset :
    # the data, shuffled and split between train and test sets
    if not os.path.exists(path):
        os.makedirs(path)

    dataset_module = datasets[dataset]
    (x_train, y_train), (x_test, y_test), input_shape, labels = dataset_module.load_data(path)

    if dataformat == 'NCHW':
        x_train,x_test=x_train.transpose([0,3,1,2]),x_test.transpose([0,3,1,2])
    elif dataformat == "NHWC":
        pass #already in this format
    else:
        raise ValueError("Invalid channel format %s" % dataformat)

    num_classes=len(labels)
    # convert class vectors to binary class matrices
    #y_train = to_categorical(y_train, num_classes)
    #y_test  = to_categorical(y_test, num_classes)

    return ClassificationDataset(dataset, x_train, x_test, y_train, y_test, num_classes, np.array(input_shape), labels,dataformat)
