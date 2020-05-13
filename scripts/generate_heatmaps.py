#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from transformation_measure import MeasureResult
import config
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import List
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from transformation_measure import visualization

import transformation_measure as tm

# def plot_last_layers_per_class(results,folderpath):
#
#     for result, model, dataset, conv_aggregation in results:
#         var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
#         combination_folderpath=os.path.join(folderpath,
#                                             variance.plots_folder(model, dataset, conv_aggregation))
#         os.makedirs(combination_folderpath,exist_ok=True)
#         plot_last_layer(var,f"{conv_aggregation}_unrotated",model,dataset,combination_folderpath)
#         plot_last_layer(rotated_var,f"{conv_aggregation}_rotated",model,dataset,combination_folderpath)
#
#
# def plot_last_layer(class_measures,training,model,dataset,folderpath):
#     classes=len(class_measures)
#     f=plt.figure()
#     variance_heatmap=np.zeros( (classes,classes) )
#
#     for i,class_measure in enumerate(class_measures):
#         variance_heatmap[:,i]=class_measure[-1]
#
#     vmax=np.nanmax(variance_heatmap)
#     mappable=plt.imshow(variance_heatmap,vmin=0,vmax=vmax,cmap='inferno',aspect="auto")
#
#     plt.xticks(range(classes))
#     plt.yticks(range(classes))
#
#     cbar = f.colorbar(mappable, extend='max')
#     cbar.cmap.set_over('green')
#     cbar.cmap.set_bad(color='blue')
#     plt.tight_layout()
#
#     image_path= os.path.join(folderpath,f"last_layer_{training}.png")
#
#     plt.savefig(image_path)
#     plt.close()
#
#
# def plot_heatmaps(results:List[variance.VarianceExperimentResult]):
#
#     measure_results=[r.measure_result for r in results]
#     vmin, vmax = visualization.outlier_range_all(measure_results, iqr_away=3)
#     vmin=0
#     folderpath= variance.plots_base_folder()
#     for r in results:
#         detail=f"{r.id()}"
#         name=f"{detail}.png"
#         stratified_name = f"{detail}_stratified.png"
#         visualization.plot_heatmap(detail, r.measure_result.numpy.id(), r.measure_result.activation_names, vmin=vmin, vmax=vmax, savefig=folderpath, savefig_name=name)

heatmaps_folder=config.heatmaps_folder()
heatmaps_folder.mkdir(exist_ok=True,parents=True)

folder = config.results_folder()
for f in sorted(folder.iterdir()):
    if not f.is_file():
        continue
    result=config.load_result(f)
    if "savepoint=" in result.parameters.model_name():
        continue
    if "rep=" in result.parameters.model_name():
        continue
    full_model_name=result.parameters.model_name()
    index = full_model_name.index("(")
    model_name = full_model_name[:index]
    model_folderpath = heatmaps_folder / model_name / full_model_name
    model_folderpath.mkdir(exist_ok=True,parents=True)
    filepath= model_folderpath / f"{result.id()}.jpg"
    visualization.plot_heatmap(result.measure_result,filepath)

    measure_result= result.measure_result
    if measure_result.__class__ == tm.StratifiedMeasureResult:
        stratified_folderpath = model_folderpath / f"{result.id()}_stratified"
        stratified_folderpath.mkdir(exist_ok=True,parents=True)
        measure_result:tm.StratifiedMeasureResult = measure_result
        n = len(measure_result.class_labels)
        for i,class_name,measure in zip(range(n),measure_result.class_labels,measure_result.class_measures):
            title = f"{i:02}_{class_name}"
            filepath = stratified_folderpath / f"{title}.jpg"
            visualization.plot_heatmap(measure,filepath)



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