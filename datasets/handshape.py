import handshape_datasets as hd

from .util import reduce_size_subset_stratified
from pathlib import Path
import numpy as np


class HandshapeLoader:
    def __init__(self, id:str, train_percent=0.8, min_samples_per_class=None):
        self.id=id
        self.train_percent=train_percent
        self.min_samples_per_class=min_samples_per_class

    def load_data(self,path:Path):
        info = hd.info(self.id)
        x,metadata = hd.load(self.id,path)
        y=metadata["y"]
        x,y=self.remove_classes_with_insufficient_samples(x,y)


        x_train,y_train,x_test,y_test=reduce_size_subset_stratified(self.train_percent, x, y, random=False)
        # print(y_train.max())
        return (x_train, y_train), (x_test, y_test), info.input_shape,info.labels

    def remove_classes_with_insufficient_samples(self,x,y):
        if self.min_samples_per_class is None:
            return x,y
        # count samples per class
        (unique, counts) = np.unique(y, return_counts=True)
        # find indices of samples whose class

        indices=np.array([False]*x.shape[0])
        for i,c in zip(unique,counts):
            if c >= self.min_samples_per_class:
                sample_indices_for_class=y==i
                # sample_indices_for_class=sample_indices_for_class[0]
                indices=np.logical_or(indices,sample_indices_for_class)
                # indices = np.hstack([indices,sample_indices_for_class])
        # print(indices.shape,indices.dtype)
        x=x[indices,:]
        y=y[indices]
        old_classes = np.sort(np.unique(y))
        for i,old_class in enumerate(old_classes):
            y[y==old_class]=i

        return x,y


    