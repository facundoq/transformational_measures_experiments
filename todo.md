# Run stuff
* stratified
* all DM 
* compare models
* feature map visualization
* BN
* train vs test
* SimplestConv. When finished, replace SimpleConv if all OK.
* random model
    
# Fix stuff
* Show model scores obtained with the savepoints for the Training experiment 
* DM and DSEM where using manhattan distances
    * DSEM didn't have normalized layers
    * Rerun all measures
* NM used no aggregation => lots of errors
    * Check if all ok
    * talk about this in conv agg analysis
* Check wtf is wrong with NM + mnist when not using any conv agg => values are too high!
* Check wtf is wrong with kernel 7x7 not learning mnist r16
    * It seems with r8 it learns slowly, and its normal for r2/r0
* Debug Goodfellow's method
* Check anova many runtime errors => fixed with conv agg? test
* Check SimplestConv on cifar10

          
# Experiments
* Stratified
* Low priority
    * Invariance vs number of layers
    * Retraining: get a previously trained model and retrain with another set of transformations. Use only one measure (NM/Anova)
        * Vanilla => Affine
        * Affine => Vanilla
        * Rotation => Other Affine (and viceversa)
        * Rotation => More rotation angles
        * Rotation => Less rotation angles
    * Train without invariance, then train with invariance to X, then test invariance to Y in both models. Does invariance to one thing helps in invariance to another?
    * Retraining experiments
        * Which layers get the invariance now?
    * Different activation functions (ELU vs ReLU vs pReLU vs TanH)
    * Invariance to X vs epochs needed to train (use results) (ignore for now)
    * Train invariance to X in cifar with 5 classes, then test with other 5 classes. Is the invariance still there?
# Measures    

* Robust measures
* Multivariate tests
* Normalized measure with statistical significance
* Implement Goodfellow's method
    * https://ai.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf
    * Approximate percentile https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf
* Implement equivalence testing
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5856600/
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533933/
    * https://stats.stackexchange.com/questions/430575/anova-like-equivalence-testing-to-measure-invariance
    * https://en.wikipedia.org/wiki/Uniformly_most_powerful_test
* Anova
    * Holm-Bonferroni correction for Anova (https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)
    
       
# Extra models      
* Spatial Transformers 
    * https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
* Ti POOLING: rotar img, evaluar, luego pooling sobre feature maps finales antes de fc1
    * implementar en pytorch
    * https://github.com/dlaptev/TI-pooling/blob/master/torch/rot_mnist12K_model.lua
    * https://github.com/dlaptev/TI-pooling
* GrouPy
    * https://github.com/adambielski/pytorch-gconv-experiments
* DREN:Deep Rotation Equivirant Network
    * https://github.com/ZJULearning/DREN
* Polar transformer
    * https://github.com/daniilidis-group/polar-transformer-networks/blob/master/arch.py
* deform conv
    * https://github.com/4uiiurz1/pytorch-deform-conv-v2
    * https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch
* ORN
    * https://github.com/ZhouYanzhao/ORN
    * torch    
* Gabor convolutional networks
    * https://github.com/bczhangbczhang/Gabor-Convolutional-Networks
    * torch
* Rotation equivariant vector field networks
    * https://arxiv.org/abs/1612.09346
*   Learning Steerable Filters for Rotation Equivariant CNNs
    * https://zpascal.net/cvpr2018/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.pdf


# Misc
* Convert cmd arguments from fixed set of choices to separate options. IMPORTANT!!!!
    * Make each parameters object implement a get_parser() to get the ArgParser for the parameter, so that they can be reused in different scripts.
    * They should also implement a to_cmd() method, that generates the command line string representation, so that the runners can create Parameter objects and then just use to_cmd() to call the other scripts
    * Also centralize somewhere the map from a parameter object to a filename which contains the result.




