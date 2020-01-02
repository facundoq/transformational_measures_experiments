#!/usr/bin/env bash
export OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export KMP_INIT_AT_FORK=FALSE
python3 -m cProfile -o profile.txt -s tottime "$@"
#tuna profile.txt
#gprof2dot -f pstats profile.txt -o profile.dot
#dot -Tpng profile.dot -o profile.png


