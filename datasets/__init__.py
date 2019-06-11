
from . import cluttered_mnist,mnist_rot,mnist,fashion_mnist,cifar10
from typing import List

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
    def __init__(self,name:str,
                 x_train:np.ndarray,x_test:np.ndarray,y_train:np.ndarray,y_test:np.ndarray,
                 num_classes:int,input_shape:np.ndarray,labels:List[str],dataformat:str):
        self.name=name
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.num_classes=num_classes
        self.input_shape=input_shape
        self.labels=labels
        self.dataformat=dataformat

    def reduce_size_subset(self,percentage,x,y):
        n=len(y)
        
        for i in range(self.num_classes):
            class_indices= y==i

            indices = np.random.permutation(n)
        indices = indices[:int(n * percentage)]
        x = self.x_train[indices, :]
        y = self.y_train[indices, :]
        return x,y

    def reduce_size(self,percentage):
        x_train,y_train=self.reduce_size_subset(percentage,self.x_train,self.y_train)
        x_test, y_test = self.reduce_size_subset(percentage, self.x_test, self.y_test)


        return ClassificationDataset(self.name,x_train,x_test, y_train,y_test
                                     ,self.num_classes,self.input_shape,self.labels,self.dataformat)

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
