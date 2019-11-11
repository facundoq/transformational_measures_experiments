from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator

from multiprocessing import Queue
import threading
import abc

class LayerMeasure(abc.ABC):
    def __init__(self,id:int,name:str):
        self.id=id
        self.name=name

    @abc.abstractmethod
    def eval(self,q:Queue,inner_q:Queue):
        pass

    @abc.abstractmethod
    def get_final_result(self):
        pass

    def queue_as_generator(self,q: Queue):
        while True:
            v = q.get()
            if v is None:
                break
            else:
                yield v

from enum import Enum

class ActivationsOrder(Enum):
    TransformationsFirst = "tf"
    SamplesFirst = "sf"

class PerLayerMeasure(Measure,abc.ABC):

    def __init__(self,activations_order:ActivationsOrder):
        self.activations_order = activations_order
    @abc.abstractmethod
    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        pass

    def signal_iteration_end(self, queues: [Queue]):
        self.put_value(queues,None)

    def wait_for_threads(self, threads: [threading.Thread]):
        for t in threads:
            t.join()

    def put_values(self,queues: [Queue],values:[]):
        for q,v in zip(queues,values):
            q.put(v)

    def put_value(self,queues:[Queue],value):
        for q in queues:
            q.put(value)

    def start_threads(self,threads:[threading.Thread]):
        for t in threads:
            t.start()

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        names = activations_iterator.activation_names()
        layers = len(names)
        layer_measures = [self.generate_layer_measure(i, name) for i, name in enumerate(names)]
        queues = [Queue() for i in range(layers)]
        inner_queues = [Queue() for i in range(layers)]

        threads = [threading.Thread(target=c.eval, args=[q, qi],daemon=True) for c, q, qi in
                   zip(layer_measures, queues, inner_queues)]

        self.start_threads(threads)
        if self.activations_order == ActivationsOrder.SamplesFirst:
            self.eval_samples_first(activations_iterator,queues,inner_queues)
        elif self.activations_order == ActivationsOrder.TransformationsFirst:
            self.eval_transformations_first(activations_iterator, queues, inner_queues)
        else:
            raise ValueError(f"Unknown activations order {self.activations_order}")

        self.wait_for_threads(threads)
        results = [r.get_final_result() for r in layer_measures]
        return MeasureResult(results, names, self)

    def eval_samples_first(self,activations_iterator:ActivationsIterator, queues:[Queue], inner_queues:[Queue]):

        for activations, x_transformed in activations_iterator.samples_first():
            self.put_value(queues, x_transformed)
            self.put_values(inner_queues,activations)
            self.signal_iteration_end(inner_queues)
        self.signal_iteration_end(queues)


    def eval_transformations_first(self, activations_iterator: ActivationsIterator, queues: [Queue],inner_queues: [Queue]):

        for transformation, batch_activations in activations_iterator.transformations_first():
            self.put_value(queues,transformation)
            for x, batch_activation in batch_activations:
                self.put_values(inner_queues,batch_activation)
            self.signal_iteration_end(inner_queues)
        self.signal_iteration_end(queues)


class SamplesFirstPerLayerMeasure(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.SamplesFirst)

class TransformationsFirstPerLayerMeasure(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.TransformationsFirst)