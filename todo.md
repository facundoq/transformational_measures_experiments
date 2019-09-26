* Check wtf is wrong with NM + mnist when not using any conv agg => values are too high!
* Anova
    *DONE:  Bonferroni correction for Anova
    * Holm-Bonferroni correction for Anova (https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)
* Define more experiment configurations
    * DONE: Invariance vs epochs trained, ¿for all datasets/models? (how to implement?)
    * DONE:  Stratified vs none -> mnist/cifar, simple_conv/resnet
    * DONE: Invariance to rotation: train with n rotations, test with more.
    * DONE: Transformation strength vs invariance obtained
    * DONE: Invariance to random networks 
    * Which model is more invariant (plot invariance for each model resampling x axis (layer))
    * DONE: Train with dataset x, measure invariance with dataset Y
    * Invariance to X vs epochs needed to train (use results)
    * Train invariance to X in cifar with 5 classes, then test with other 5 classes. Is the invariance still there?
        
    * After refactoring model parser  
        * Invariance vs number of layers
        * Different activation functions (ELU vs ReLU vs pReLU vs TanH)
        * Retraining: get a previously trained model and retrain with another set of transformations. Use only one measure (NM/Anova)
            * Vanilla => Affine
            * Affine => Vanilla
            * Rotation => Other Affine (and viceversa)
            * Rotation => More rotation angles
            * Rotation => Less rotation angles
        * Train without invariance, then train with invariance to X, then test invariance to Y in both models. Does invariance to one thing helps in invariance to another?
    * Retraining experiments
        * Which layers get the invariance now?
        

* Convert cmd arguments from fixed set of choices to separate options. IMPORTANT!!!!
    * Make each parameters object implement a get_parser() to get the ArgParser for the parameter, so that they can be reused in different scripts.
    * They should also implement a to_cmd() method, that generates the command line string representation, so that the runners can create Parameter objects and then just use to_cmd() to call the other scripts
    * Also centralize somewhere the map from a parameter object to a filename which contains the result.



