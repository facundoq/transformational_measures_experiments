
from . import cluttered_mnist,mnist_rot,mnist,fashion_mnist,cifar10


import numpy as np
import os
names=["mnist","fashion_mnist","cifar10","mnist_rot","cluttered_mnist","lsa16","pugeault"]

def get(dataset="mnist",dataformat="NCHW",path=os.path.expanduser("~/.datasets/")):
    # the data, shuffled and split between train and test sets
    if not os.path.exists(path):
        os.makedirs(path)
    if dataset=="mnist":
        (x_train, y_train), (x_test, y_test), input_shape,  labels=mnist.load_data(path)
    elif dataset=="fashion_mnist":
        (x_train, y_train), (x_test, y_test), input_shape, labels = fashion_mnist.load_data(path)
    elif dataset=="cifar10":
        (x_train, y_train), (x_test, y_test), input_shape, labels = cifar10.load_data(path)
    elif dataset == "mnist_rot":
        x_train, x_test,y_train, y_test, input_shape,labels = mnist_rot.load_data(path)
    elif dataset=="cluttered_mnist":
        (x_train, y_train), (x_test, y_test), input_shape, labels= cluttered_mnist.load_data(path)
    else:
        raise ValueError("Unknown dataset: %s" % dataset)

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

    return ClassificationDataset(dataset, x_train, x_test, y_train, y_test, num_classes, input_shape, labels,dataformat)


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class ClassificationDataset:
    def __init__(self,name,x_train,x_test,y_train,y_test,num_classes,input_shape,labels,dataformat):
        self.name=name
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.num_classes=num_classes
        self.input_shape=input_shape
        self.labels=labels
        self.dataformat=dataformat

    def summary(self):
        result=f"Image Classification Dataset {self.name}\n" \
            f"Dataformat {self.dataformat}\n"
        result+=f"x_train: {self.x_train.shape}, {self.x_train.dtype}\n"
        result+=f"x_test: {self.x_test.shape}, {self.x_test.dtype}\n"
        result+=f"y_train: {self.y_train.shape}, {self.y_train.dtype}\n"
        result+=f"y_test: {self.y_test.shape}, {self.y_test.dtype}\n"
        result+=f"Classes {np.unique(self.y_train)}\n"
        result+=f"min class/max class: {self.y_train.min()} {self.y_train.max()}"
        return result
