
from . import cluttered_mnist,mnist_rot,mnist,fashion_mnist,cifar10


import numpy as np
import os
names=["mnist","fashion_mnist","cifar10","mnist_rot","cluttered_mnist","lsa16","pugeault"]

datasets={"mnist":mnist
          ,"fashion_mnist":fashion_mnist
          ,"cifar10":cifar10
          ,"mnist_rot":mnist_rot
          ,"cluttered_mnist":cluttered_mnist
          }

def get(dataset,dataformat="NCHW",path=os.path.expanduser("~/.datasets/")):
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

    return ClassificationDataset(dataset, x_train, x_test, y_train, y_test, num_classes, input_shape, labels,dataformat)



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
