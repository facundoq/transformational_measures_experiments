
from models.all_conv import AllConvolutional
from .base import InvarianceExperiment
from ..common import *
from ..models import *

simple_models_generators = [SimpleConvConfig]
common_models_generators = [
    SimpleConvConfig,
    AllConvolutionalConfig,
    VGG16DConfig,
    ResNetConfig,
]
small_models_generators = [SimpleConvConfig ]


# ca_none, ca_mean, ca_sum, ca_max = tm.numpy.IdentityTransformation(), tm.numpy.AggregateConvolutions(), tm.numpy.AggregateTransformation(tm.numpy.AggregateFunction.sum),  tm.numpy.AggregateTransformation(tm.numpy.AggregateFunction.max)
# da = tm.numpy.DistanceAggregation(normalize=False, keep_shape=False)
# da_normalize = tm.numpy.DistanceAggregation(normalize=True, keep_shape=False)
# da_normalize_keep = tm.numpy.DistanceAggregation(normalize=True, keep_shape=True)
# da_keep = tm.numpy.DistanceAggregation(normalize=False, keep_shape=True)

# df = tm.numpy.DistanceFunction(normalize=False)
# df_normalize = tm.numpy.DistanceFunction(normalize=True)


#gf = tm.numpy.GoodfellowNormalInvariance(alpha=0.99)
#anova = tm.numpy.ANOVAInvariance(alpha=0.99,bonferroni=True)



def get_ylim_normalized(measure: tm.pytorch.PyTorchMeasure):
    # TODO dict
    if measure.__class__ == tm.pytorch.NormalizedVarianceInvariance:
        return 1.4
    elif measure.__class__ == tm.pytorch.NormalizedVarianceSameEquivariance:
        return 8
    # elif measure.__class__ == tm.pytorch.NormalizedDistanceSameEquivariance:
    #     return 8
    # elif measure.__class__ == tm.pytorch.NormalizedDistanceInvariance:
    #     return 1.4
    else:
        raise ValueError(measure)
