import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from variance_measure.measures.result import MeasureResult

def plot_heatmap(title, m:MeasureResult, layer_names, vmin=0, vmax=None, savefig=None, savefig_name=None):


    n = len(layer_names)
    f, axes = plt.subplots(1, n, dpi=150)

    for i, (activation, name) in enumerate(zip(m.layers, layer_names)):


        ax = axes[i]
        ax.axis("off")
        activation = activation[:, np.newaxis]
        if vmax is not None:
            mappable=ax.imshow(activation,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")
        else:
            mappable = ax.imshow(activation, vmin=vmin, cmap='inferno', aspect="auto")
        #mappable = ax.imshow(cv, cmap='inferno')
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
    if not savefig is None:
        image_name=f"{savefig_name}.png"
        path=os.path.join(savefig,image_name)
        plt.savefig(path)
    #plt.show()
    plt.close()

def pearson_outlier_range(values,iqr_away):
    p50 = np.median(values)
    p75 = np.percentile(values, 75)
    p25 = np.percentile(values, 25)
    iqr = p75 - p25

    range = (p50 - iqr_away * iqr, p50 + iqr_away * iqr)
    return range

def outlier_range_all(std_list,iqr_away=5):
    var_values=[np.hstack([np.hstack(class_stds) for class_stds in stds]) for stds in std_list]
    var_values=np.hstack(var_values)
    return outlier_range_values(var_values,iqr_away)

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

def plot(all_stds,model,dataset_name,savefig=False,savefig_suffix="",class_names=None,vmax=None):
    vmin=0
    classes=len(all_stds)
    for i,c in enumerate(range(classes)):
        stds=all_stds[i]
        if class_names:
            name=class_names[c]
        else:
            name=str(c)
        plot_heatmap(i, name, stds, vmin, vmax, model.activation_names(), model.name,
                     dataset_name, savefig,
                     savefig_suffix)

def plot_collapsing_layers(rotated_measures,measures,labels,savefig=None, savefig_suffix=""):
    rotated_measures_collapsed=collapse_measure_layers(rotated_measures)
    measures_collapsed=collapse_measure_layers(measures)
    n=len(rotated_measures)
    assert( n == len(measures))
    assert (n == len(labels))
    if n==2:
        color = np.array([[1,0,0],[0,0,1]])
    else:
        color = plt.cm.hsv(np.linspace(0.1, 0.9, n))


    f,ax=plt.subplots(dpi=min(350,max(150,n*15)))
    for rotated_measure,measure,label,i in zip(rotated_measures_collapsed,measures_collapsed,labels,range(n)):
        x_rotated=np.arange(rotated_measure.shape[0])
        x=np.arange(measure.shape[0])
        ax.plot(x_rotated,rotated_measure,label="rotated_"+label,linestyle="-",color=color[i,:])
        ax.plot(x,measure,label="unrotated_"+label,linestyle="--",color=color[i,:]*0.7)
        ax.set_ylabel("Variance")
        ax.set_xlabel("Layer")
        # ax.set_ylim(max_measure)
        n_layers=len(x)
        if n_layers<25:
            ax.set_xticks(range(n_layers))

    #handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    # lgd=ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 1))
    #           # mode="expand", borderaxespad=0.3)
    # # ax.autoscale_view()
    # f.artists.append(lgd) # Here's the change
    #plt.tight_layout()

    if savefig:
        image_name=f"collapsed_{savefig_suffix}.png"
        path=os.path.join(savefig,image_name)
        plt.savefig(path,bbox_inches="tight")

    #plt.show()
    plt.close()




def collapse_measure_layers(measures):
    return [np.array([np.mean(layer[np.isfinite(layer)]) for layer in measure]) for measure in measures]

def plot_heatmaps(model,rotated_model,dataset,results,folderpath):
    var, stratified_layer_vars, var_all_dataset, rotated_var, rotated_stratified_layer_vars, rotated_var_all_dataset = results
    vmin, vmax = outlier_range_all(results, iqr_away=3)
    vmin = vmin_all = vmin_class = 0
    vmax_all = vmax_class = vmax
    # vmin_class, vmax_class = outlier_range_both(rotated_var, var)
    # vmin_class = 0
    # vmin_class, vmax_class = outlier_range_both(rotated_stratified_layer_vars, stratified_layer_vars)
    # vmin_class = 0
    # vmin_all, vmax_all = outlier_range_both(rotated_var, var)

    plot(rotated_var, model, dataset.name, savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class, class_names=dataset.labels)
    plot(var, model, dataset.name, savefig=folderpath, savefig_suffix="unrotated",
         vmax=vmax_class, class_names=dataset.labels)
    plot(rotated_stratified_layer_vars, rotated_model, dataset.name,
         class_names=["all_stratified"], savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class)
    plot(stratified_layer_vars, model, dataset.name, class_names=["all_stratified"],
         savefig=folderpath, savefig_suffix="unrotated", vmax=vmax_class)

    plot(rotated_var_all_dataset, rotated_model, dataset.name,
         savefig=folderpath, savefig_suffix="rotated", class_names=["all"], vmax=vmax_all)
    plot(var_all_dataset, model, dataset.name,
         savefig=folderpath, savefig_suffix="unrotated", class_names=["all"], vmax=vmax_all)


def plot_all(model,rotated_model,dataset,results):

    folderpath=plots_folder(model.name,dataset.name)

    var, stratified_layer_vars, var_all_dataset, rotated_var, rotated_stratified_layer_vars, rotated_var_all_dataset=results

    plot_heatmaps(model, rotated_model, dataset, results,folderpath)

    # print("plotting layers invariance (by classes)")
    # plot_collapsing_layers(rotated_var, var, dataset.labels
    #                        , savefig=folderpath, savefig_suffix="classes")

    # max_rotated = max([m.max() for m in rotated_measures_collapsed])
    # max_unrotated = max([m.max() for m in measures_collapsed])
    # max_measure = max([max_rotated, max_unrotated])
    #print("plotting layers invariance (global)")
    r=results

    plot_collapsing_layers(rotated_stratified_layer_vars+rotated_var_all_dataset, stratified_layer_vars+var_all_dataset
                            , ["stratified","all"], savefig=folderpath, savefig_suffix="global")
