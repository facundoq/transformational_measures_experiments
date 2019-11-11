from .base import Measure,MeasureResult
import numpy as np
from .layer_transformation import ConvAggregation
from transformation_measure.measure.stats_running import RunningMeanAndVariance,RunningMean
import transformation_measure as tm
from multiprocessing import Queue
from transformation_measure import MeasureFunction

from .multithreaded_layer_measure import LayerMeasure,PerLayerMeasure,ActivationsOrder

class GlobalVariance(LayerMeasure):

    def eval(self,q:Queue,inner_q:Queue):
        inner_m = RunningMeanAndVariance()
        self.m = RunningMean()
        for iteration_info in self.queue_as_generator(q):
            for activations in self.queue_as_generator(inner_q):
                activations = self.preprocess_activations(activations)
                for j in range(activations.shape[0]):
                    inner_m.update(activations[j,])
            inner_result=self.measure_function.apply_running(inner_m)
            self.m.update(inner_result)

    def get_final_result(self):
        return self.m.mean()


class GlobalFiringRate(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.TransformationsFirst)
    def __repr__(self):
        return f"G()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        return GlobalVariance(i, name)

class LocalFiringRate(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.SamplesFirst)
    def __repr__(self):
        return f"L()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        return GlobalVariance(i, name)

class Goodfellow(tm.QuotientMeasure):

    def __init__(self,):
        sm = GlobalFiringRate()
        ttm = LocalFiringRate()
        super().__init__(ttm,sm)
        self.numerator_measure = ttm
        self.denominator_measure = sm

    def __repr__(self):
        return f"Goodfellow()"