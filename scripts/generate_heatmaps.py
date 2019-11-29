#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import config
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import List
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from transformation_measure import visualization
import typing
from experiment import variance
import os,sys
import argparse,argcomplete
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
#         visualization.plot_heatmap(detail, r.measure_result.measure.id(), r.measure_result.activation_names, vmin=vmin, vmax=vmax, savefig=folderpath, savefig_name=name)

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
    model_folderpath = heatmaps_folder / result.parameters.model_name()
    model_folderpath.mkdir(exist_ok=True,parents=True)
    filepath= model_folderpath / f"{result.id()}.png"
    visualization.plot_heatmap(result.measure_result,filepath,"Heatmap")

    #if result.parameters.stratified:
    measure_result= result.measure_result
    if measure_result.__class__ == tm.StratifiedMeasureResult:
        stratified_folderpath = model_folderpath / f"{result.id()}_stratified"
        stratified_folderpath.mkdir(exist_ok=True,parents=True)
        measure_result:tm.StratifiedMeasureResult = measure_result
        n = len(measure_result.class_labels)
        for i,class_name,measure in zip(range(n),measure_result.class_labels,measure_result.class_measures):
            title = f"{i:02}_{class_name}"
            filepath = stratified_folderpath / f"{title}.png"
            visualization.plot_heatmap(measure,filepath,title)





