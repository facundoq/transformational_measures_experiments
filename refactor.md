
* Rerun all experiments
    * IMPORTANT BEFORE
        * Restore dataset percentage from 0.01 to 0.5
        * Restore training epochs from 1 to formula
        * Restore min accuracy from 0 to formula
* Split measures for every framework
    * Measures for PyTorch/TF receive Model, Dataset and Transformation objects
    * Measures for Numpy still receive an Iterator adapter. 
    * Transformations are specific to each framework now.
* fix numpy transformations
* see TODOs

* Add tqdm for training
    * by epochs, not batches
* Add tqdm for measures
    * Apply in iterator
    * Datasets need to have a __len__ property



* Add more examples and documentation to lib
* Reference papers in code

LISTO:

* Split Language, make a custom one for the lib
* Redefine Transformations in terms of intensity and density 
* Remove "set input shape" from
 transformations
    * Reimplement PyTorch AffineTransformation without using numpy     
* Fix measure experiment code
* Fix experiments that define custom transformations
* Fix naming issues and bugs