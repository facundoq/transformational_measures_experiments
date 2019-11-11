* layer based measure
    * reimplement ANOVA
    * add multiprocessing and test times
* implement https://ai.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf
    * with layer based measure
* fix *intensity* of transformations, make finite limits
    * translation should be exp (0,1,2,4,8,16)
        * Express in % of image moved
        * 0%, 1%, 2%, 4%, 8%, 16%
            * check how to integrate both dims, ie, -2,0 equal to -1,-1
    * scale should be more relevant (too similar for some cases)
        * lower limit, upper limit: 50%, 125%
        * start from 100% 
            * decrement 10%, increment 5%
            * 0: 100%, => 1: 90%, 105% => 2: 80%, 110% => ... => 5: 50%, 125%          
    * rotation should be incremental from 0 (no rotation) to 50 (-90  to 90 deg) to 100 (full -180 to 180 deg). Keep number of transformations the same? make proportional to strength? Yes, up to 16 transformations.



