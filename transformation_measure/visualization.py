params = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],
    # 'image.origin': 'lower',
    # 'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    # 'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    # 'figure.figsize': [3.39, 2.10],
    'font.family': 'sans',
}
import matplotlib
matplotlib.rcParams.update(params)

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from pytorch.numpy_dataset import NumpyDataset
from transformation_measure.measure.base import MeasureResult
import transformation_measure as tm
from pathlib import Path
from experiment import variance



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




def plot_collapsing_layers_different_models(results:List[variance.VarianceExperimentResult], filepath:Path, labels=None,title="",linestyles=None,color=None,legend_location=None):

    n = len(results)
    if n == 0:
        raise ValueError(f"`results` is an empty list.")
    if color is None:
        if n==2:
            color = np.array([[1,0,0],[0,0,1]])
        else:
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))

    f, ax = plt.subplots(dpi=min(350, max(150, n * 15)))
    f.suptitle(title)

    result_layers = np.array([len(r.measure_result.layer_names) for r in results])
    min_n, max_n = result_layers.min(), result_layers.max()
    x_result_most_layers=np.zeros(1)
    for i, result in enumerate(results):
        n_layers = len(result.measure_result.layers)
        x = np.linspace(0,100,n_layers,endpoint=True)
        if n_layers>=x_result_most_layers.size:
            x_result_most_layers=x
        y = result.measure_result.per_layer_average()

        if labels is None:
            label = None
        else:
            label = labels[i]
        if linestyles is None:
            linestyle = "-"
        else:
            linestyle = linestyles[i]
        ax.plot(x, y, label=label, linestyle=linestyle, color=color[i, :] * 0.7)

    ax.set_ylabel("Measure values")
    ax.set_xlabel("Layer (%)")

    x_result_most_layers_int = x_result_most_layers.astype(int)
    ax.set_xticks(x_result_most_layers_int)
    x_result_most_layers_str = [str(x) for x in x_result_most_layers_int]
    ax.set_xticklabels(x_result_most_layers_str,rotation=45)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put legend below current axis
    if legend_location is None:
        loc, pos = ['lower center', np.array((0.5, 0))]
    else:
        loc, pos = legend_location

    ax.legend(loc=loc, bbox_to_anchor=pos,
              fancybox=True, shadow=True)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_collapsing_layers_same_model(results:List[variance.VarianceExperimentResult], filepath:Path, labels=None,title="",linestyles=None,plot_mean=False,color=None,legend_location=None):
    n=len(results)
    if n == 0:
        raise ValueError(f"`results` is an empty list.")
    if color is None:
        if n==2:
            color = np.array([[1,0,0],[0,0,1]])
        else:
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))

    f, ax = plt.subplots(dpi=min(350, max(150, n * 15)))
    f.suptitle(title)

    result_layers=np.array([len(r.measure_result.layer_names) for r in results])
    min_n,max_n = result_layers.min(),result_layers.max()
    if plot_mean:
        assert min_n==max_n,"To plot the mean values all results must have the same number of layers."

    average=np.zeros(max_n)

    for i, result in enumerate(results):
        n_layers= len(result.measure_result.layers)

        x= np.arange(n_layers)+1
        y= result.measure_result.per_layer_average()
        if plot_mean:
            average+=y
        if labels is None:
            label = None
        else:
            label = labels[i]
        if linestyles is None:
            linestyle="-"
        else:
            linestyle=linestyles[i]
        ax.plot(x,y,label=label,linestyle=linestyle,color=color[i,:]*0.7)


    if plot_mean:
        x = np.arange(max_n)+1
        y=average/len(results)
        label="mean"
        linestyle="--"
        ax.plot(x, y, label=label, linewidth=3, linestyle=linestyle, color=(0,0,0))

    ax.set_ylabel("Variance")
    ax.set_xlabel("Layer")
    # ax.set_ylim(max_measure)
    if max_n < 32:
        labels = results[0].measure_result.layer_names
        x = np.arange(max_n) + 1
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put legend below current axis
    if legend_location is None:
        loc, pos = ['lower center', np.array((0.5, 0))]
    else:
        loc, pos = legend_location

    ax.legend(loc=loc, bbox_to_anchor=pos,
              fancybox=True, shadow=True)

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

import datasets
import torch

import typing


from pathlib import Path

def plot_activations(features:np.ndarray, feature_indices:[int], variance_scores:[float], x_transformed:np.ndarray, transformations:tm.TransformationSet, filepath:Path):
    x_transformed= x_transformed.transpose((0, 2, 3, 1))
    n,c = features.shape
    nt,ht,wt,ct = x_transformed.shape

    dpi = int(np.sqrt(n) * 9) + 150
    f, axis = plt.subplots(c+1,n+1,dpi=dpi)
    axis[0, 0].set_title("Inputs")
    axis[0, 0].axis("off")

    for i in range(c):
        axis[i+1, 0].set_title(f"Act {feature_indices[i]} \n, score:{variance_scores[i]:0.2f} ",fontsize=5)
        axis[i+1, 0].axis("off")

    for j in range(n):
        transformed_image= x_transformed[j, :]
        transformed_image-=transformed_image.min()
        transformed_image/= transformed_image.max()
        if ct ==1:
            transformed_image=transformed_image[:,:,0]
        axis[0,j+1].imshow(transformed_image,cmap="gray")
        axis[0,j+1].axis("off")

    for i in range(c):
        mi, ma = features[:,i].min(), features[:,i].max()
        for j in range(n):
            image = features[j, i].reshape((1,1))
            axis[i+1,j+1].imshow(image, vmin=mi, vmax=ma,cmap="gray")
            axis[i+1, j+1].axis("off")

    plt.savefig(filepath)
    plt.close("all")

def plot_feature_maps(feature_maps:np.ndarray, feature_indices:[int], variance_scores:[float], x_transformed:np.ndarray, transformations:tm.TransformationSet, filepath:Path):
    feature_maps = feature_maps.transpose((0,2,3,1))
    x_transformed= x_transformed.transpose((0, 2, 3, 1))
    n,h,w,c = feature_maps.shape
    nt,ht,wt,ct = x_transformed.shape
    if ct == 1:
        x_transformed = x_transformed[:,:, :, 0]

    fontsize=max(3, 10 - int(np.sqrt(n)))
    dpi = int(np.sqrt(n) * 9) + 100
    f, axis = plt.subplots(c+1, n+4, dpi=dpi)
    axis[0, 0].set_title("Inputs")
    axis[0, 0].axis("off")
    for i in range(c):
        title=f"FM {feature_indices[i]}\n, score:\n{variance_scores[i]:0.2f} "
        axis[i+1, 0].text(0,0, title,fontsize=fontsize)
        axis[i+1, 0].axis("off")

    for j in range(n):
        transformed_image  = x_transformed[j, :]
        transformed_image -= transformed_image.min()
        transformed_image /= transformed_image.max()

        axis[0,j+1].imshow(transformed_image)
        axis[0,j+1].axis("off")

    colorbar_images=[]
    for i in range(c):
        mi, ma = feature_maps[:,:,:,i].min(), feature_maps[:,:,:,i].max()
        for j in range(n):
            im=axis[i+1,j+1].imshow(feature_maps[j,:,:,i],vmin=mi,vmax=ma,cmap="gray")
            axis[i+1, j+1].axis("off")
            if j+1 == n:
                colorbar_images.append(im)

    # mean and std of feature maps columns
    for i in range(c):
        mean_feature_map = np.mean(feature_maps[:,:,:,i],axis=0)
        std_feature_map = np.std(feature_maps[:, :, :, i], axis=0)
        axis[i + 1, -3].imshow(mean_feature_map,cmap="gray")
        axis[i + 1, -3].axis("off")
        axis[i + 1, -2].imshow(std_feature_map,cmap="gray")
        axis[i + 1, -2].axis("off")

    for i in range(c):
        axis[i + 1, -1].axis("off")
        cbar = plt.colorbar(colorbar_images[i], ax=axis[i + 1, -1])
        cbar.ax.tick_params(labelsize=fontsize)
    axis[0, -1].axis("off")

        # mean and std of images columns
    axis[0, -3].imshow(x_transformed.mean(axis=0),cmap="gray")
    axis[0, -3].axis("off")

    axis[0, -2].imshow(x_transformed.std(axis=0),cmap="gray")
    axis[0, -2].axis("off")

    plt.savefig(filepath)
    plt.close("all")

def indices_of_smallest_k(a,k):

    indices = np.argpartition(a, k)[:k]
    values = a[indices]
    indices = np.argsort(a)

    # ind.sort()
    return indices[:k]

def indices_of_largest_k(a,k):

    # indices = np.argpartition(a, -k)[:k]
    # values = a[indices]
    indices=np.argsort(a)

    # ind.sort()
    return indices[-k:]

def select_feature_maps(measure_result:tm.MeasureResult, most_invariant_k:int,least_invariant_k:int):
    feature_indices_per_layer=[]
    feature_scores_per_layer = []
    values=measure_result.layers
    layer_names=measure_result.layer_names
    for value,name in zip(values,layer_names):
        if len(value.shape) != 1:
            raise ValueError("Feature maps should be collapsed before calling this function")
        most_invariant_indices = indices_of_smallest_k(value, most_invariant_k)
        least_invariant_indices = indices_of_largest_k(value, least_invariant_k)
        indices = np.concatenate([most_invariant_indices,least_invariant_indices])
        feature_indices_per_layer.append(indices)
        feature_scores_per_layer.append(value[indices])
    return feature_indices_per_layer,feature_scores_per_layer

'''
    Plots the activation of the invariant feature maps determined by :param result
    Plots the best :param features_per_layer feature maps, ie the most invariant
    Creates a plot for each sample image/transformation pair
    '''

def plot_invariant_feature_maps(plot_folderpath:Path, activations_iterator:tm.ActivationsIterator, result:variance.VarianceExperimentResult,  most_invariant_k:int,least_invariant_k:int, conv_aggregation:tm.ConvAggregation):
    measure_result=result.measure_result.collapse_convolutions(conv_aggregation)

    feature_indices_per_layer,invariance_scores_per_layer=select_feature_maps(measure_result, most_invariant_k,least_invariant_k)
    transformations=activations_iterator.get_transformations()

    for i_image,(layers_activations, x_transformed) in enumerate(activations_iterator.samples_first()):
        for i_layer,layer_activations in enumerate(layers_activations):
            layer_name=result.measure_result.layer_names[i_layer]
            filepath = plot_folderpath / f"image{i_image}_layer{i_layer}_{layer_name}.png"
            feature_indices=feature_indices_per_layer[i_layer]
            invariance_scores=invariance_scores_per_layer[i_layer]
            #only plot conv layer activations
            if len(layer_activations.shape)==4:
                feature_maps=layer_activations[:,feature_indices,:,:]
                plot_feature_maps(feature_maps,feature_indices,invariance_scores,x_transformed,transformations,filepath)
            elif len(layer_activations.shape)==2:
                features = layer_activations[:, feature_indices]
                plot_activations(features, feature_indices, invariance_scores, x_transformed, transformations,filepath)



def plot_invariant_feature_maps_pytorch(plot_folderpath:Path,model:torch.nn.Module,dataset:datasets.ClassificationDataset,transformations:tm.TransformationSet,result:variance.VarianceExperimentResult,images=8,most_invariant_k:int=4,least_invariant_k:int=4,conv_aggregation=tm.ConvAggregation.mean):
    numpy_dataset = NumpyDataset(dataset.x_test[:images,:],dataset.y_test[:images])
    iterator = tm.PytorchActivationsIterator(model,numpy_dataset, transformations, batch_size=images)
    plot_invariant_feature_maps(plot_folderpath,iterator,result,most_invariant_k,least_invariant_k,conv_aggregation)


