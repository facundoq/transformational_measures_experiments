from .base import Measure,MeasureResult,ActivationsByLayer
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np


def divide_activations(a:ActivationsByLayer, b:ActivationsByLayer)->ActivationsByLayer:
        eps = 0
        measures = []  # coefficient of variations

        for layer_v_transformations,layer_v_samples in zip(a, b):
            # print(layer_baseline.shape, layer_measure.shape)
            normalized_measure = layer_v_transformations.copy()
            normalized_measure[layer_v_samples  > eps] /= layer_v_samples [layer_v_samples  > eps]
            both_below_eps = np.logical_and(layer_v_samples  <= eps,
                                            layer_v_transformations <= eps)
            normalized_measure[both_below_eps] = 1
            only_baseline_below_eps = np.logical_and(
                layer_v_samples  <= eps,
                layer_v_transformations > eps)
            normalized_measure[only_baseline_below_eps] = np.inf
            measures.append(normalized_measure)
        return measures

class QuotientMeasure(Measure):
    def __init__(self, numerator_measure:Measure,denominator_measure:Measure):
        super().__init__()
        self.numerator_measure=numerator_measure
        self.denominator_measure=denominator_measure

    def __repr__(self):
        return f"QM({self.numerator_measure}_DIV_{self.denominator_measure})"

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names = activations_iterator.activation_names()
        v_samples=self.denominator_measure.eval(activations_iterator)

        v_transformations=self.numerator_measure.eval(activations_iterator)

        v=divide_activations(v_transformations.layers, v_samples.layers)
        return MeasureResult(v,layer_names,self)


