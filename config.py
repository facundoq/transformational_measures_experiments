import os,pickle



def base_path():
    return os.path.expanduser("~/variance/")



from experiment import training

def model_path(p: training.Parameters):
    model_folderpath= models_folder()
    filename=f"{p.id()}.pt"
    filepath=os.path.join(model_folderpath,filename)
    return filepath

def models_folder():
    model_folderpath = os.path.join(base_path(), "models")
    os.makedirs(model_folderpath, exist_ok=True)
    return model_folderpath

def get_models_filenames():
    files=os.listdir(models_folder())
    model_filenames=[f for f in files if f.endswith(".pt")]
    return model_filenames
def get_models_filepaths():
    model_folderpath = models_folder()
    return [os.path.join(model_folderpath,f) for f in get_models_filenames()]

def training_plots_path():
    plots_folderpath = "training_plots"
    plots_folderpath = os.path.join(base_path(), plots_folderpath)
    os.makedirs(plots_folderpath, exist_ok=True)
    return plots_folderpath




def variance_results_folder()->str:
    return os.path.join(base_path(), "results")

from experiment import variance

def results_path(p:variance.Parameters, results_folder=variance_results_folder()):

    return  os.path.join(results_folder, f"{p.id()}.pickle")

def save_results(r:variance.VarianceExperimentResult, results_folder=variance_results_folder()):
    path = results_path(r.parameters, results_folder)
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_result(path)->variance.VarianceExperimentResult:
    return pickle.load(open(path, "rb"))


def load_results(filepaths:[str])-> [variance.VarianceExperimentResult]:
    results = []
    for filepath in filepaths:
        result = load_result(filepath)
        results.append(result)
    return results

def load_all_results(folderpath:str)-> [variance.VarianceExperimentResult]:
    filepaths=[os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]
    return load_results(filepaths)


def plots_base_folder():
    return os.path.join(base_path(), "plots")

# def plots_folder(r:VarianceExperimentResult):
#     folderpath = os.path.join(plots_base_folder(), f"{r.id()}")
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath,exist_ok=True)
#     return folderpath
