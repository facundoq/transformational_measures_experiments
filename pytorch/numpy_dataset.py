from torch.utils.data import Dataset

class NumpyDataset(Dataset):

    def __init__(self, *data_sources):
        ''''''

        assert(len(data_sources)>0)
        self.data_sources=data_sources
        lengths=[ d.shape[0] for d in self.data_sources]
        assert(check_equal(lengths))

    def __len__(self):
        return self.data_sources[0].shape[0]

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def get_batch(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return ( d[idx,:] for d in self.data_sources)

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
