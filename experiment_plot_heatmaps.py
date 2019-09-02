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

if __name__ == '__main__':
    results_folder=config.variance_results_folder()
    results_paths=[os.path.join(results_folder,f) for f in os.listdir(results_folder)]
    results_paths=filter(lambda x: x.endswith(".pickle"),results_paths)
    results= config.load_results(results_paths)
    folderpath=os.path.join(config.plots_base_folder(),"heatmaps")
    os.makedirs(folderpath,exist_ok=True)
    for result in results:
        filepath=os.path.join(folderpath,f"{result.id()}.png")
        visualization.plot_heatmap(result.measure_result, filepath=filepath, title=str(result))