* Fix measure experiment code
* Fix experiments that define custom transformations
* Split measures for every framework
    * Measures for PyTorch/TF receive Model, Dataset and Transformation objects
    * Measures for Numpy still receive an Iterator adapter. 
    * Transformations are specific to each framework now.
* see TODOs

LISTO:

* Split Language, make a custom one for the lib
* Redefine Transformations in terms of intensity and density 
* Remove "set input shape" from
 transformations
    * Reimplement PyTorch AffineTransformation without using numpy     