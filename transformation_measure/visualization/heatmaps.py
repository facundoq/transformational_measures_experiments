import numpy as np
import matplotlib.pyplot as plt
from typing import List
from transformation_measure.measure.base import MeasureResult
import transformation_measure as tm
from pathlib import Path


def plot_heatmap(m:MeasureResult,filepath:Path,title:str, vmin=0, vmax=None):

    # for l in m.layers:
    #     print(l.shape, np.sum(np.isinf(l)))

    m=m.collapse_convolutions(tm.ConvAggregation.mean)

    # for l in m.layers:
    #     print(l.shape, np.sum(np.isinf(l)))

    n = len(m.layer_names)
    f, axes = plt.subplots(1, n, dpi=150)
    for i, (activation, name) in enumerate(zip(m.layers, m.layer_names)):
        ax = axes[i]
        ax.axis("off")
        activation = activation[:, np.newaxis]
        #mappable = ax.imshow(cv, cmap='inferno')
        if vmax is not None:
            mappable = ax.imshow(activation,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")
        else:
            mappable = ax.imshow(activation, vmin=vmin, cmap='inferno', aspect="auto")

        if n<40:
            if len(name)>6:
                name=name[:6]
            ax.set_title(name, fontsize=4,rotation = 45)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    # f.suptitle(f"{title}", fontsize=10)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    plt.savefig(filepath)
    plt.close()

def pearson_outlier_range(values,iqr_away):
    p50 = np.median(values)
    p75 = np.percentile(values, 75)
    p25 = np.percentile(values, 25)
    iqr = p75 - p25

    range = (p50 - iqr_away * iqr, p50 + iqr_away * iqr)
    return range

def outlier_range_all(results:List[MeasureResult],iqr_away=5):
    all_values = []
    for r in results:
        for layer in r.layers():
            all_values.append(layer[:])
    all_values = np.hstack(all_values)

    #var_values=[np.hstack([np.hstack(values) for values in stds]) for stds in std_list]

    return outlier_range_values(all_values,iqr_away)

    # minmaxs=[outlier_range(stds,iqr_away) for stds in std_list]
    # mins,maxs=zip(*minmaxs)
    # return max(mins),min(maxs)

def outlier_range_both(rotated_stds,unrotated_stds,iqr_away=5):
    rmin,rmax=outlier_range(rotated_stds,iqr_away)
    umin,umax= outlier_range(unrotated_stds,iqr_away)

    return (max(rmin,umin),min(rmax,umax))

def outlier_range_values(values,iqr_away):
    pmin, pmax = pearson_outlier_range(values, iqr_away)
    # if the pearson outlier range is away from the max and/or min, use max/or and min instead

    finite_values=values[np.isfinite(values)]
    # print(pmax, finite_values.max())
    return (max(pmin, finite_values.min()), min(pmax, finite_values.max()))

def outlier_range(stds,iqr_away):
    class_values=[np.hstack(class_stds) for class_stds in stds]
    values=np.hstack(class_values)

    return outlier_range_values(values,iqr_away)