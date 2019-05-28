import os
import pickle

class VarianceExperimentResult:
    def __init__(self, model_name, dataset_name, activation_names, class_names, transformations, options, rotated_measures, unrotated_measures):
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.activation_names=activation_names
        self.class_names=class_names
        self.transformations=transformations
        self.options=options
        self.rotated_measures=rotated_measures
        self.unrotated_measures=unrotated_measures
    def description(self):
        description = "-".join([str(v) for v in self.options.values()])
        return description


results_folder=os.path.expanduser("~/variance_results/values")

def get_path(model_name,dataset_name,description):
    return os.path.join(results_folder, f"{model_name}_{dataset_name}_{description}.pickle")


def save_results(r:VarianceExperimentResult):
    path=get_path(r.model_name,r.dataset_name,r.description())
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_results(path)->VarianceExperimentResult:
    return pickle.load(open(path, "rb"))

def plots_base_folder():
    return os.path.expanduser("~/variance_results/plots/")
    #return os.path.join(results_folder,"plots/var")

def plots_folder(r:VarianceExperimentResult):
    folderpath = os.path.join(plots_base_folder(), f"{r.model_name}_{r.dataset_name}_{r.description()}")

    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath