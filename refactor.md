* Copy code from invariance experiments to same_equivariance experiments

* Add parameter to modelconfig to indicate if should add classification head

* Modify training for SE to use parameters as Y
    * Reencode parameters? rotation as (cos,sin)
* Adapt trainer to use MSE as loss function
    * Adopt Poutyne for training to see if can speed up transformations 
* Train models for same_equivariance
    * Check performance
* Modify same_equivariance to use SE measures

* Split measures for every framework
    * Measures for PyTorch/TF receive Model, Dataset and Transformation objects
    * Measures for Numpy still receive an Iterator adapter. 
    * Transformations are specific to each framework now.
* fix numpy transformations
* see TODOs
  
* Add more examples and documentation to lib
* Reference papers in code
