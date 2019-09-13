import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

from transformation_measure.measure.base import MeasureResult
import transformation_measure as tm
def plot_heatmap(m:MeasureResult,filepath:str,title:str, vmin=0, vmax=None):

    m=m.collapse_convolutions(tm.ConvAggregation.sum)

    n = len(m.layer_names)
    f, axes = plt.subplots(1, n, dpi=150)
    for i, (activation, name) in enumerate(zip(m.layers, m.layer_names)):
        ax = axes[i]
        ax.axis("off")
        activation = activation[:, np.newaxis]
        #mappable = ax.imshow(cv, cmap='inferno')
        if vmax is not None:
            mappable=ax.imshow(activation,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")
        else:
            mappable = ax.imshow(activation, vmin=vmin, cmap='inferno', aspect="auto")

        if n<40:
            if len(name)>6:
                name=name[:6]
            ax.set_title(name, fontsize=4)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"{title}", fontsize=10)
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



from experiment import variance


def plot_collapsing_layers(results:List[variance.VarianceExperimentResult], filepath, labels=None,title="",linestyles=None):
    n=len(results)
    if n==2:
        color = np.array([[1,0,0],[0,0,1]])
    else:
        color = plt.cm.hsv(np.linspace(0.1, 0.9, n))


    f,ax=plt.subplots(dpi=min(350,max(150,n*15)))
    f.suptitle(title)
    for i, result in enumerate(results):
        n_layers= len(result.measure_result.layers)
        x= np.arange(n_layers)+1
        y= result.measure_result.per_layer_average()
        if labels is None:
            label=result.parameters.id()
        else:
            label=labels[i]
        if linestyles is None:
            linestyle="-"
        else:
            linestyle=linestyles[i]
        ax.plot(x,y,label=label,linestyle=linestyle,color=color[i,:]*0.7)
        ax.set_ylabel("Variance")
        ax.set_xlabel("Layer")
        # ax.set_ylim(max_measure)
        if n_layers<25:
            ax.set_xticks(range(n_layers))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True)

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


