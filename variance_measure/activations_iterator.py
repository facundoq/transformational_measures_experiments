import abc

class ActivationsIterator:

    """
    Iterate over the activations of a network, varying the samples and transformations
    In both orders
    """
    def __init__(self, model, dataset, transformations):
        self.model=model
        self.dataset=dataset
        self.transformations=transformations

    @abc.abstractmethod
    def transformations_first(self):
        pass

    @abc.abstractmethod
    def samples_first(self):
        pass