import os
import typing

import datasets
import transformation_measure as tm
from enum import Enum

class DatasetSubset(Enum):
    train="train"
    test="test"
    values=[train,test]

class DatasetParameters:
    def __init__(self,name:str,subset: DatasetSubset,percentage:float):
        assert(percentage>0)
        assert(percentage<=1)
        self.name=name
        self.subset=subset
        self.percentage=percentage
    def __repr__(self):
        return f"{self.name}({self.subset.value},p={self.percentage:.2})"
    def id(self):
        return str(self)

class Parameters:
    def __init__(self, model_path:str, dataset:DatasetParameters, transformations:tm.TransformationSet, measure:tm.Measure,stratified:bool=False):
        self.model_path=model_path
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations
        self.stratified=stratified
    def model_name(self):
        base,filename_ext=os.path.split(self.model_path)
        filename,ext=os.path.splitext(filename_ext)
        return filename
    def id(self):
        measure=self.measure.id()
        if self.stratified:
            measure=f"Stratified({measure})"
        return f"{self.model_name()}_{self.dataset}_{self.transformations.id()}_{measure}"

    def __repr__(self):
        measure = self.measure.id()
        if self.stratified:
            measure = f"Stratified({measure})"
        return f"VarianceExperiment parameters: models={self.model_name()}, dataset={self.dataset} transformations={self.transformations}, measure={measure}"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int,adapt_dataset:bool):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.adapt_dataset=adapt_dataset

class VarianceExperimentResult:
    def __init__(self, parameters:Parameters, measure_result:tm.MeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result

    def __repr__(self):
        description = f"VarianceExperimentResult, params: {self.parameters}"
        return description

    def id(self):
        return f"{self.parameters.id()}"


dataset_names=["mnist","cifar10"]

import config

def possible_experiment_parameters()->[]:
    transformations = config.all_transformations(10)
    measures= config.all_measures()

    dataset_percentages = config.common_dataset_sizes()
    dataset_subsets=[DatasetSubset.train,DatasetSubset.test]
    datasets=[]
    for dataset in dataset_names:
        for dataset_subset in dataset_subsets:
            for dataset_percentage in dataset_percentages:
                datasets.append(DatasetParameters(dataset,dataset_subset,dataset_percentage))

    parameters=[datasets, measures, transformations]
    def list2dict(list):
        return {x.id(): x for x in list}
    parameters=[ list2dict(p) for p in parameters ]

    return parameters

import argcomplete, argparse


def parse_parameters()->typing.Tuple[Parameters,Options]:
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])

    def is_valid_file(filepath):
        if not os.path.exists(filepath):
            raise argparse.ArgumentTypeError("The model file %s does not exist!" % filepath)
        else:
            return filepath

    datasets, measures, transformations=possible_experiment_parameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", metavar="mo",type=is_valid_file,required=True)
    parser.add_argument("-dataset", metavar="d", choices=datasets.keys(),required=True)
    parser.add_argument("-measure", metavar="me", choices=measures.keys(),required=True)
    parser.add_argument("-stratified", metavar="stra",type=bool_parser,default=False)
    parser.add_argument("-adapt_dataset", metavar="adapt", type=bool_parser, default=False)
    parser.add_argument("-transformation", metavar="t", choices=transformations.keys(),required=True)
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/models/transformations')

    parser.add_argument('-num_workers', metavar='nw'
                        , help=f'num_workersto use during training'
                        , type=int
                        , default=2)
    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during eval'
                        , type=int
                        , default=256)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()


    p = Parameters(args.model,
                   datasets[args.dataset],
                   transformations[args.transformation],
                   measures[args.measure],stratified=args.stratified)
    o = Options(args.verbose,args.batchsize,args.num_workers,args.adapt_dataset)
    return p,o





import numpy as np
import cv2

def expand_channels(dataset:datasets.ClassificationDataset,c:int):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1

    dataset.x_train = np.repeat(dataset.x_train,c,axis=axis)
    dataset.x_test = np.repeat(dataset.x_test, c, axis=axis)


def collapse_channels(dataset:datasets.ClassificationDataset):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1
    dataset.x_train = dataset.x_train.mean(axis=axis,keepdims=True)
    dataset.x_test  = dataset.x_test.mean(axis=axis,keepdims=True)


def resize(dataset:datasets.ClassificationDataset,h:int,w:int,c:int):

    if dataset.dataformat=="NCHW":
        dataset.x_train=np.transpose(dataset.x_train,axes=(0,2,3,1))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 2, 3, 1))

    subsets = [dataset.x_train, dataset.x_test]
    new_subsets=[np.zeros((s.shape[0],h,w,c)) for s in subsets]

    for (subset,new_subset) in zip(subsets,new_subsets):
        for i in range(subset.shape[0]):
            img=subset[i, :]
            if c==1:
                #remove channel axis, resize, put again
                img=img[:,:,0]
                img= cv2.resize(img, dsize=(h, w))
                img = img[:, :, np.newaxis]
            else:
                #resize
                img = cv2.resize(img, dsize=(h, w))

            new_subset[i,:]=img

    dataset.x_train = new_subsets[0]
    dataset.x_test = new_subsets[1]

    if dataset.dataformat=="NCHW":
        dataset.x_train = np.transpose(dataset.x_train,axes=(0,3,1,2))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 3, 1, 2))

def adapt_dataset(dataset:datasets.ClassificationDataset, dataset_template:str):
    dataset_template = datasets.get(dataset_template)
    h,w,c= dataset_template.input_shape
    del dataset_template
    oh,ow,oc=dataset.input_shape

    # fix channels
    if c !=oc and oc==1:
        expand_channels(dataset,c)

    elif c != oc and c ==1:
        collapse_channels(dataset)
    else:
        raise ValueError(f"Cannot transform image with {oc} channels into image with {c} channels.")

    #fix size
    if h!=oh or w!=ow:
        resize(dataset,h,w,c)

    dataset.input_shape=(h,w,c)