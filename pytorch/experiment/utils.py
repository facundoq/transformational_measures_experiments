

import argparse,argcomplete
import pytorch.experiment.model_loading as models
import datasets
import transformations

def parse_model_and_dataset(description):

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(models.get_model_names())}'
                        , choices=models.get_model_names())
    parser.add_argument('dataset', metavar='d',
                        help=f'Dataset to train/eval model. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names)
    parser.add_argument('transformation', metavar='t',nargs='+',
                        help=f'Transformations to apply to the dataset to train a model. Allowed values: {", ".join(transformations.names)}'
                        , choices=transformations.names)
    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    print(args.transformation)
    return args.model,args.dataset,args.transformation
