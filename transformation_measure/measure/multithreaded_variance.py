from .base import Measure,MeasureResult
import numpy as np
from .layer_transformation import ConvAggregation
from transformation_measure.measure.stats_running import RunningMeanAndVariance,RunningMean
import transformation_measure as tm
from multiprocessing import Queue
from transformation_measure import MeasureFunction

from .multithreaded_layer_measure import LayerMeasure,PerLayerMeasure,ActivationsOrder

class VarianceLayerMeasure(LayerMeasure):

    def __init__(self,id:int,name:str,measure_function: MeasureFunction,conv_aggregation:ConvAggregation):
        super().__init__(id,name)
        self.conv_aggregation=conv_aggregation
        self.measure_function=measure_function

    def eval(self,q:Queue,inner_q:Queue):

        self.m = RunningMean()
        for iteration_info in self.queue_as_generator(q):
            inner_m = RunningMeanAndVariance()
            for activations in self.queue_as_generator(inner_q):
                activations = self.conv_aggregation.apply(activations)
                for j in range(activations.shape[0]):
                    inner_m.update(activations[j,])
            inner_result=self.measure_function.apply_running(inner_m)
            self.m.update(inner_result)

    def get_final_result(self):
        return self.m.mean()

class VarianceMeasure(PerLayerMeasure):
    def __init__(self,order:ActivationsOrder,measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__(order)
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        return VarianceLayerMeasure(i,name,self.measure_function,self.conv_aggregation)

class TransformationVarianceMeasure(VarianceMeasure):
    def __init__(self,measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__(ActivationsOrder.SamplesFirst,measure_function,conv_aggregation)
    def __repr__(self):
        return f"TVM(f={self.measure_function.value},ca={self.conv_aggregation.value})"

class SampleVarianceMeasure(VarianceMeasure):
    def __init__(self,measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__(ActivationsOrder.TransformationsFirst,measure_function,conv_aggregation)

    def __repr__(self):
        return f"SVM(f={self.measure_function.value},ca={self.conv_aggregation.value})"

class NormalizedVarianceMeasure(tm.QuotientMeasure):

    def __init__(self,measure_function: MeasureFunction, conv_aggregation:ConvAggregation):
        sm = SampleVarianceMeasure(measure_function, conv_aggregation)
        ttm = TransformationVarianceMeasure(measure_function, conv_aggregation)
        super().__init__(ttm,sm)
        self.numerator_measure = ttm
        self.denominator_measure = sm
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def __repr__(self):
        return f"NVM(f={self.measure_function.value},ca={self.conv_aggregation.value})"