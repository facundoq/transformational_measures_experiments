from torch.utils.data import Dataset
import torch
import numpy as np
def check_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)



class NumpyDataset(Dataset):

    @classmethod
    def stratify_dataset(cls,y,*data_sources):
        '''

        :param y: class labels
        :param data_sources: list of numpy arrays. The first dim of each array must match len(y)
        :return: a list of NumpyDatasets, one for each class in np.unique(y), with the samples corresponding to each class
        '''
        for d in data_sources:
            assert(d.shape[0]==y.shape[0])
        classes = np.unique(y)
        classes.sort()

        per_class_variance = []
        # calculate the var measure for each class
        iterators=[]
        for i, c in enumerate(classes):
            # logging.debug(f"Evaluating vars for class {c}...")
            ids = np.where(y == c)
            ids = ids[0]
            data_sources_class=[ x[ids, :] for x in data_sources ]
            y_class = y[ids]
            data_sources_class.append(y_class)
            iterators.append(NumpyDataset(*data_sources_class))
        return iterators

    def __init__(self, *data_sources):
        assert(len(data_sources)>0)
        self.data_sources=data_sources
        lengths=[ d.shape[0] for d in self.data_sources]
        assert(check_equal(lengths))

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def __len__(self):
        return self.data_sources[0].shape[0]

    def get_batch(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        batch=( torch.from_numpy(d[idx,]) for d in self.data_sources)
        return batch

    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)






class NumpyKeyValueDataset(Dataset):

    def __init__(self, **data_sources):
        '''  '''

        assert(len(data_sources)>0)
        self.data_sources=data_sources
        lengths=[ d.shape[0] for d in self.data_sources.values()]
        assert(check_equal(lengths))

    def __len__(self):
        return self.data_sources.values()[0].shape[0]

    def __getitem__(self, idx):
        #return ( (key,d[idx,:]) for key,d in self.data_sources.items())
        return self.get_batch(idx)

    def get_batch(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return ( (key,d[idx,:]) for key,d in self.data_sources.items())

    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)

