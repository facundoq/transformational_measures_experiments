import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..language import l

from transformational_measures.visualization import default_discrete_colormap

def plot_accuracies(plot_filepath:Path, accuracies_by_label:[[float]], labels:[str], group_names:[str]):
    # set width of bar
    f=plt.figure(dpi=300)
    patterns = ["...","**","\\\\\\","///",  "+" , "x", "o", "O", ".", "*" ,"/" , "\\" , "|" , "-" ,]

    accuracies_by_label= accuracies_by_label.T
    n_groups=len(group_names)
    n_labels=len(labels)
    barWidth = 1/(n_labels+1)
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    pos = np.arange(n_groups,dtype=float)
    pad = barWidth*0.1
    for i,(accuracies,label) in enumerate(zip(accuracies_by_label, labels)):
        if n_labels <= len(patterns):
            hatch=patterns[i]
        else:
            hatch=None
    # Make the plot
        plt.bar(pos, accuracies, color=cmap(i), width=barWidth, edgecolor='white', label=label,hatch=hatch)
        pos+=barWidth+pad
    plt.gca().set_ylim(0,1)

    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.model)
    plt.ylabel(l.accuracy)
    def shorten(label:str): return label if len(label)<=10  else label[:9]+"."

    group_names = [shorten(label) for label in group_names]

    plt.xticks([r + barWidth for r in range(len(group_names))], group_names)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath,bbox_inches='tight')
    plt.close()

def plot_metrics_single_model(plot_filepath:Path, metrics:[float], labels:[str], metric="accuracy"):
    # set width of bar
    f=plt.figure(dpi=300)
    n=len(labels)
    assert len(metrics) == n, f"Different number of labels {n} and {metric} {len(metrics)} "
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    x = np.arange(n,dtype=float)
    colors = np.array([cmap(i) for i in range(n)])
    # print(colors)
    # Make the plot
    print(metrics,labels)
    plt.bar(x, metrics, color=colors, edgecolor='white')
    if metric == "accuracy":
        plt.gca().set_ylim(0,1)
    # elif metric in ["rmse" , "mse", "mae","mape"]:
    if metric == "rae":
        ma = max(metrics)
        ma = max(ma,1.1)
        plt.gca().set_ylim(0, ma)
    else:
        mi,ma=min(metrics),max(metrics)
        mi = 0
        delta=0.1
        plt.gca().set_ylim(mi*delta,ma*(1+delta))


    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.transformation)
    plt.ylabel(metric)

    plt.xticks(x, labels)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath,bbox_inches='tight')
    plt.close()


