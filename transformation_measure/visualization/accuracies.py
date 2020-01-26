import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.language import l

from . import default_discrete_colormap

def plot_accuracies(plot_filepath:Path,accuracies_by_transformation:[[float]],transformation_names:[str],model_names:[str]):
    # set width of bar
    f=plt.figure(dpi=300)
    n_models=len(model_names)
    n_transformations=len(transformation_names)
    barWidth = 1/(n_transformations+1)
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    pos = np.arange(n_models,dtype=float)
    for i,(accuracies,transformation_name) in enumerate(zip(accuracies_by_transformation,transformation_names)):
    # Make the plot
        plt.bar(pos, accuracies, color=cmap(i), width=barWidth, edgecolor='white', label=transformation_name)
        pos+=barWidth
    plt.gca().set_ylim(0,1)

    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.model)
    plt.ylabel(l.accuracy)
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath)
    plt.close()

def plot_accuracies_single_model(plot_filepath:Path,accuracies:[float],transformation_names:[str]):
    # set width of bar
    f=plt.figure(dpi=300)

    n=len(transformation_names)
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    x = np.arange(n,dtype=float)
    colors = np.array([cmap(i) for i in range(n)])
    # print(colors)
    # Make the plot
    plt.bar(x, accuracies,color=colors, edgecolor='white')

    plt.gca().set_ylim(0,1)

    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.transformation)
    plt.ylabel(l.accuracy)

    plt.xticks(x, transformation_names)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath)
    plt.close()


