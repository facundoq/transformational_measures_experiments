
# Main entry points

* `invariance.py` runs invariance measure experiments
* `same_equivariance_run.py` runs same equivariance measure experiments


# Folders

* config: general experiment configurations (deprecated?)
* datasets: Dataset loading code
* experiment: General experiment code
* experiments: actual experiments
    * same_equivariance: same equivariance experiments
    * invariance: invariance experiments
    * tasks: generic train/eval tasks
    * test
    * visualization: common visualization code 
* testing: unit tests for datasets, activation iterators, and transformations
* scripts: useful scripts to view results and generate plots for paper
* pytorch: specific pytorch trianing/loading code, as well as common regression metrics (MAE, etc)
* models: models for invariance (not equivariance)