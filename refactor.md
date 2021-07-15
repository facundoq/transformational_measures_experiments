
* Change rotation parameter so that it's naturally from 0 to 1 and therefore combined measures have a controlled range
* Modify same_equivariance to use SE measures

* Split measures for every framework
    * Measures for PyTorch/TF receive Model, Dataset and Transformation objects
    * Measures for Numpy still receive an Iterator adapter. 
    * Transformations are specific to each framework now.
* fix numpy transformations
* see TODOs
  
* Add more examples and documentation to lib
* Reference papers in code
