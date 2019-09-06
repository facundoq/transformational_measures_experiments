import os,pickle
from experiment import variance


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



def results_paths(ps:[variance.Parameters], results_folder=variance_results_folder()):
    variance_paths= [f'{results_path(p)}' for p in ps]
    return variance_paths

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




from transformation_measure import *

def common_measures()-> [Measure]:
    measures=[ SampleMeasure(MeasureFunction.std, ConvAggregation.sum)
             ,TransformationMeasure(MeasureFunction.std, ConvAggregation.sum)
     ,NormalizedMeasure(MeasureFunction.std, ConvAggregation.sum)
        ,AnovaFMeasure(ConvAggregation.none)
        ,AnovaMeasure(ConvAggregation.none,alpha=0.05)

    ]
    return measures

def common_transformations() -> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(n_rotations=16)
        , SimpleAffineTransformationGenerator(n_translations=2)
        , SimpleAffineTransformationGenerator(n_scales=2)]
    return transformations

def rotation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_rotations=2**r) for r in range(1,n+1)]

def scale_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_scales=2**r) for r in range(n)]

def translation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_translations=r) for r in range(1,n+1)]

def all_transformations(n:int):
    return common_transformations()+rotation_transformations(n)+scale_transformations(n)+translation_transformations(n)



def common_dataset_sizes()->[float]:
    return [0.01,0.02,0.05,0.1,0.5,1.0]
