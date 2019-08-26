#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
from transformation_measure import visualization
import typing
import config
import os,sys
import argparse,argcomplete





def parse_parameters()->(str,typing.List[str],bool):
    # results_folderpath = variance.default_results_folder()
    # results_files=os.listdir(results_folderpath)
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument('results', type=str, nargs='+')
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/model/transformations')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    return args.name,args.out,args.results,args.verbose

if __name__ == '__main__':
    experiment_name, output_filepath, results_paths, verbose=parse_parameters()
    for result_path in results_paths:
        if not os.path.exists(result_path):
            print(f"Path to non-existent result file {result_path}")
            sys.exit()
    results= config.load_results(results_paths)
    visualization.plot_collapsing_layers(results, filepath=output_filepath, experiment_name=experiment_name)