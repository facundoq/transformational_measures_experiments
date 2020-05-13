from . import cluttered_mnist,mnist_rot,mnist,fashion_mnist,cifar10,handshape
from typing import List
from datasets.util import reduce_size_subset_stratified
import numpy as np
import os
from enum import Enum
from pathlib import Path

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


    def normalize_features(self):
        self.x_test=self.x_test.astype(np.float32)
        self.x_train=self.x_train.astype(np.float32)

        def mean_std(x:np.ndarray):
            u=x.mean(axis=(0,2,3),keepdims=True)
            d=x.std(axis=(0,2,3),keepdims=True)
            d[d==0]=1
            return u,d
        def normalize(x,u,d):
            x-=u
            x/=d
        u,d=mean_std(self.x_train)
        normalize(self.x_train,u,d)
        normalize(self.x_test,u,d)

    def reduce_size_stratified(self, percentage):
        if percentage==1:
            return self
        x_train, y_train, _, _ = reduce_size_subset_stratified(percentage, self.x_train, self.y_train)
        x_test, y_test, _, _ = reduce_size_subset_stratified(percentage, self.x_test, self.y_test)
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


datasets={"mnist":mnist,
          "lsa16":handshape.HandshapeLoader("lsa16"),
          "rwth":handshape.HandshapeLoader("rwth", min_samples_per_class=15),
          # ,"fashion_mnist":fashion_mnist
          "cifar10":cifar10,
          # ,"mnist_rot":mnist_rot
          # ,"cluttered_mnist":cluttered_mnist
          }
names=datasets.keys()


def get(dataset,dataformat="NCHW",path=Path("~/.datasets/").expanduser()) -> ClassificationDataset :
    # the data, shuffled and split between train and test sets
    path.mkdir(exist_ok=True,parents=True)

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
