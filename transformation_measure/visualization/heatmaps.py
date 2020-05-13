import numpy as np
import matplotlib.pyplot as plt
from transformation_measure.numpy.base import MeasureResult
import transformation_measure as tm
from pathlib import Path

def get_limit(m:MeasureResult, op_code:str):
    ops={"max":np.max,"min":np.min}
    op = ops[op_code]

    vals = np.array([op(l) for l in m.layers])
    return op(vals)


def plot_heatmap(m:MeasureResult,filepath:Path, vmin=None, vmax=None):

    # for l in m.layers:
    #     print(l.shape, np.sum(np.isinf(l)))
    if vmax is None: vmax = get_limit(m, "max")
    if vmin is None :
        vmin = get_limit(m, "min")
        if vmin>0:
            vmin=0

    m=m.collapse_convolutions(tm.ConvAggregation.mean)

    # for l in m.layers:
    #     print(l.shape, np.sum(np.isinf(l)))

    n = len(m.layer_names)
    f, axes = plt.subplots(1, n, dpi=150)
    for i, (activation, name) in enumerate(zip(m.layers, m.layer_names)):
        # print(activation.shape,activation.min(),activation.max())
        ax = axes[i]
        ax.axis("off")
        activation = activation[:, np.newaxis]
        mappable = ax.imshow(activation,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")

        if n<40:
            if len(name)>7:
                name=name[:6]+"."
            ax.set_title(name, fontsize=4,rotation = 45)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    plt.savefig(filepath)
    plt.close()



