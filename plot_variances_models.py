import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from variance_measure import visualization, variance
import os
import datasets

def load_results(folderpath):
    results = []
    for filename in os.listdir(folderpath):
        path = os.path.join(folderpath, filename)
        model, dataset, description = visualization.get_model_and_dataset_from_path(path)
        result = visualization.load_results(model, dataset,description)
        results.append((result, model, dataset,description))
    return results


def as_table(results):
    table = {}
    for experiment_result, model, dataset, description  in results:

        for k, measure in experiment_result.rotated_measures.items():
            table[f"{dataset}_{model}_rotated__{k}_{experiment_result.options}"] = measure

        for k, measure in experiment_result.unrotated_measures.items():
            table[f"{dataset}_{model}_unrotated__{k}_{experiment_result.options}"] = measure
    return table

# def global_results_latex(results,stratified):
#     table={}
#
#     for experiment_result, model, dataset, in results:
#         var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
#         conv_table=table.get(conv_aggregation,{})
#         dataset_table=conv_table.get(dataset,{})
#
#         rotated_table=dataset_table.get("rotated",{})
#         unrotated_table = dataset_table.get("unrotated", {})
#         if stratified:
#             rotated_table[model] = variance.global_average_variance(rotated_stratified_layer_vars)
#             unrotated_table[model] = variance.global_average_variance(stratified_layer_vars)
#         else:
#             rotated_table[model]= variance.global_average_variance(rotated_var_all_dataset)
#             unrotated_table[model] = variance.global_average_variance(var_all_dataset)
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
                                            visualization.plots_folder(model, dataset, conv_aggregation))
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
        values=[f"{l:.2}" for l in measure.average_per_layer()]
        measure_string=", ".join(values)
        print(key,measure_string)




def plot_heatmaps(variance_experiment_result):

    for experiment_result, model_name, dataset_name, description  in results:
        folderpath = visualization.plots_folder(model_name, dataset_name,description)

        rotated_results=[m.layers for m in experiment_result.rotated_measures.values()]
        unrotated_results = [m.layers for m in experiment_result.unrotated_measures.values()]
        values = rotated_results+unrotated_results
        if options
        vmin, vmax = visualization.outlier_range_all(values, iqr_away=3)
        vmin=0

        for measure_name,measure in experiment_result.rotated_measures.items():
            name =f"{dataset_name}_{model_name}_rotated_{measure_name}"
            title=f"{name}\n{experiment_result.options}"
            visualization.plot_heatmap(title,measure,experiment_result.activation_names,vmin=vmin,savefig=folderpath,savefig_name=name)

        for measure_name, measure in experiment_result.unrotated_measures.items():
            name = f"{dataset_name}_{model_name}_unrotated_{measure_name}"
            title = f"{name}\n{experiment_result.options}"
            visualization.plot_heatmap(title, measure,experiment_result.activation_names,vmin=vmin, savefig=folderpath,
                                       savefig_name=name)



from pytorch.experiment.variance import VarianceExperimentResult

results_folderpath=os.path.expanduser("~/variance_results/")
results=load_results(visualization.results_folder)
table_results=as_table(results)
#print_global_results(table_results)
plot_heatmaps(results)

#plot_last_layers_per_class(results,results_folderpath)

# print("Plotting heatmaps")
# plot_all(results)
#print_global_results(results)