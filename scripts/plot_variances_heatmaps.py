#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import List
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from transformation_measure import visualization
import typing
from run import variance
import os,sys
import argparse,argcomplete


def plot_last_layers_per_class(results,folderpath):

    for result, model, dataset, conv_aggregation in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        combination_folderpath=os.path.join(folderpath,
                                            variance.plots_folder(model, dataset, conv_aggregation))
        os.makedirs(combination_folderpath,exist_ok=True)
        plot_last_layer(var,f"{conv_aggregation}_unrotated",model,dataset,combination_folderpath)
        plot_last_layer(rotated_var,f"{conv_aggregation}_rotated",model,dataset,combination_folderpath)


def plot_last_layer(class_measures,training,model,dataset,folderpath):
    classes=len(class_measures)
    f=plt.figure()
    variance_heatmap=np.zeros( (classes,classes) )

    for i,class_measure in enumerate(class_measures):
        variance_heatmap[:,i]=class_measure[-1]

    vmax=np.nanmax(variance_heatmap)
    mappable=plt.imshow(variance_heatmap,vmin=0,vmax=vmax,cmap='inferno',aspect="auto")

    plt.xticks(range(classes))
    plt.yticks(range(classes))

    cbar = f.colorbar(mappable, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    plt.tight_layout()

    image_path= os.path.join(folderpath,f"last_layer_{training}.png")

    plt.savefig(image_path)
    plt.close()


def plot_heatmaps(results:List[variance.VarianceExperimentResult]):

    measure_results=[r.measure_result for r in results]
    vmin, vmax = visualization.outlier_range_all(measure_results, iqr_away=3)
    vmin=0
    folderpath= variance.plots_base_folder()
    for r in results:
        detail=f"{r.id()}"
        name=f"{detail}.png"
        stratified_name = f"{detail}_stratified.png"
        visualization.plot_heatmap(detail, r.measure_result.measure.id(), r.measure_result.activation_names, vmin=vmin, vmax=vmax, savefig=folderpath, savefig_name=name)




def parse_parameters()->(str,typing.List[str],bool):
    # results_folderpath = variance.default_results_folder()
    # results_files=os.listdir(results_folderpath)
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name",type=str)
    parser.add_argument('results', type=str, nargs='+')
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/model/transformations')
    parser.add_argument('-verbose', metavar='v', type=bool_parser, default=True,
                        help=f'Print info about dataset/model/transformations')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    return args.experiment_name,args.results,args.verbose

experiment_name, results_paths, verbose=parse_parameters()
for result_path in results_paths:
    if not os.path.exists(result_path):
        print(f"Path to non-existent result file {result_path}")
        sys.exit()
results= variance.load_results(results_paths)

#print(experiment_name, results_paths, verbose)
plot_heatmaps(results)

#plot_last_layers_per_class(results,results_folderpath)

# print("Plotting heatmaps")
# plot_all(results)
#print_global_results(results)