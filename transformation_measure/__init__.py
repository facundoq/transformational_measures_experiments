from .transformation import TransformationSet,Transformation,IdentityTransformation
from .image_transformations import AffineTransformationGenerator,SimpleAffineTransformationGenerator,AffineTransformation, AffineTransformationCV

from .iterators.pytorch_activations_iterator import PytorchActivationsIterator,ObservableLayersModel

from .measure.base import MeasureResult,Measure,MeasureFunction,StratifiedMeasureResult
from .measure.layer_transformation import ConvAggregation
from .measure.normalized import NormalizedMeasure
from .measure.samples import SampleMeasure
from .measure.transformations import TransformationMeasure


from typing import List
def common_measures()-> List[Measure]:
    measures=[ SampleMeasure(MeasureFunction.std, ConvAggregation.sum),
             TransformationMeasure(MeasureFunction.std, ConvAggregation.sum),
     NormalizedMeasure(TransformationMeasure(MeasureFunction.std, ConvAggregation.sum), SampleMeasure(MeasureFunction.std, ConvAggregation.sum))
    ]
    return measures

def common_transformations() -> List[TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> List[TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(n_rotations=16)
        , SimpleAffineTransformationGenerator(n_translations=2)
        , SimpleAffineTransformationGenerator(n_scales=2)]
    return transformations

def rotation_transformations():
    return [SimpleAffineTransformationGenerator(n_rotations=r) for r in [1,2,4,8,16,32]]

def scale_transformations():
    return [SimpleAffineTransformationGenerator(n_scales=r) for r in [1,2,4,8,16,32]]

def translation_transformations():
    return [SimpleAffineTransformationGenerator(n_translations=r) for r in [1,2,3,4,5,6]]

def all_transformations():
    return common_transformations()+rotation_transformations()+scale_transformations()+translation_transformations()