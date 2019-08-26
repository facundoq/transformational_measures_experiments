#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import List
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from transformation_measure import visualization

from experiment import variance
import os
from experiment.variance import VarianceExperimentResult
from time import gmtime, strftime


def as_table(results:List[VarianceExperimentResult]):
    table = {}
    for r  in results:
        all={"rotated":r.rotated_measures,"unrotated":r.rotated_measures}
        for training_mode,measures in all.items():
            for k, measure in measures.items():
                table[f"{r.dataset_name}_{r.model_name}_{training_mode}_{k}_{r.options}"] = measure
    return table

# def global_results_latex(results,stratified):
#     table={}
#
#     for experiment_result, models, dataset, in results:
#         var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
#         conv_table=table.get(conv_aggregation,{})
#         dataset_table=conv_table.get(dataset,{})
#
#         rotated_table=dataset_table.get("rotated",{})
#         unrotated_table = dataset_table.get("unrotated", {})
#         if stratified:
#             rotated_table[models] = variance.global_average_variance(rotated_stratified_layer_vars)
#             unrotated_table[models] = variance.global_average_variance(stratified_layer_vars)
#         else:
#             rotated_table[models]= variance.global_average_variance(rotated_var_all_dataset)
#             unrotated_table[models] = variance.global_average_variance(var_all_dataset)
#
#         dataset_table["rotated"]=rotated_table
#         dataset_table["unrotated"] = unrotated_table
#
#         conv_table[dataset]=dataset_table
#         table[conv_aggregation]=conv_table
#
#     return table

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



#
# def format_results_latex(results_table):
#     table_str = ""
#     for conv,dataset_table in results_table.items():
#         table_str += f"conv agg {conv}\n"
#         for dataset,training_table in dataset_table.items():
#             for training,models in training_table.items():
#                 table_str += f"{dataset.upper()} & {training} & "
#                 model_keys = sorted(models.keys())
#                 table_str += " & ".join([f"{models[key]:0.2}" for key in model_keys])
#                 table_str += " \\\\ \n"
#
#     return table_str

# def print_global_results_latex(results):
#     global_results_table_latex=global_results_latex(results,stratified=False)
#     latex_table_str=format_results_latex(global_results_table_latex)
#     print("Normal results:")
#     print(latex_table_str)
#
#     stratified_global_results_table_latex=global_results_latex(results,stratified=True)
#     stratified_latex_table_str=format_results_latex(stratified_global_results_table_latex)
#     print("Stratified results:")
#     print(stratified_latex_table_str)

    # def format_results_latex(results_table):
    # table_str=""
    # for key in sorted(global_results_table.keys()):
    #     table_str += f"{key} => {global_results_table[key]}\n"
    #
    # global_results_filepath=os.path.join(variance.plots_base_folder(),"global_invariance_comparison.txt")
    # with open(global_results_filepath, "w") as text_file:
    #     text_file.write(table_str)
    #
    # print(table_str)
    #

def print_global_results(table_results):
    print("Weighted average")
    for key in sorted(table_results.keys()):
        print(key,table_results[key].weighted_global_average())
    print("Unweighted average")
    for key in sorted(table_results.keys()):
        print(key,table_results[key].global_average())
    print("Per layer, collapsed")
    for key in sorted(table_results.keys()):
        measure=table_results[key].collapse_convolutions("mean")
        values=[f"{l:.2}" for l in measure.per_layer_average()]
        measure_string=", ".join(values)
        print(key,measure_string)

def plot_heatmaps(results:List[VarianceExperimentResult]):
    timestamp=strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    for r in results:
        folderpath = variance.plots_folder(r)

        rotated_results=[m.layers for m in r.rotated_measures.values()]
        unrotated_results = [m.layers for m in r.unrotated_measures.values()]
        values = rotated_results+unrotated_results

        vmin, vmax = visualization.outlier_range_all(values, iqr_away=3)
        vmin=0
        all_results={"rotated":rotated_results,"unrotated":unrotated_results}
        for model_training,result in all_results.items():
            for measure_name,measure in r.rotated_measures.items():
                detail=f"{r.dataset_name}_{r.model_name}_{model_training}_{measure_name}"
                name =f"{model_training}_{measure_name}"
                title=f"{detail}\n{r.id()}"
                visualization.plot_heatmap(title,measure,r.activation_names,vmin=vmin,vmax=vmax,savefig=folderpath,savefig_name=name)






results_folderpath= variance.default_results_folder()
results= variance.load_results   (results_folderpath)
print(f"Found {len(results)} results, plotting..")
table_results=as_table(results)
#print_global_results(table_results)
plot_heatmaps(results)

#plot_last_layers_per_class(results,results_folderpath)

# print("Plotting heatmaps")
# plot_all(results)
#print_global_results(results)