import numpy as np
import tmeasures
from scipy.stats import f_oneway

def anova_same_mean_test(results:list[tmeasures.MeasureResult]):
    n_layers = [len(r.layers) for r in results]
    n = n_layers[0]
    assert all(map(lambda x: x==n,n_layers)),"Results should have the same number of activations"
    p_values = np.zeros(n)
    for i in range(n):
        groups = [r.layers[i].flatten() for r in results]
        f,p_value = f_oneway(*groups)
        p_values[i]=p_value
        

    return p_values.mean()