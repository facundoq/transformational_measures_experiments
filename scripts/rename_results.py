#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
from experiment import measure, training
import texttable
import config
import transformation_measure as tm
import shutil

if __name__ == '__main__':
    results_folderpath=config.results_folder()
    print(results_folderpath)
    files=results_folderpath.iterdir()

    # for f in files:
    #     if f.endswith("AnovaMeasure(ca=none).pickle"):
    #         source = os.path.join(results_folderpath,f)
    #         dest = os.path.join(results_folderpath, "old", f)
    #         shutil.move(source,dest)
    #         # f=f[:-8]
    #         # f=f+",alpha=0.99,bonferroni=False).pickle"
    #         # dest = os.path.join(results_folderpath,f)
    #         # print(f"moving:")
    #         # print(source)
    #         # print(dest)
    #         #shutil.move(source,dest)

    results=config.load_all_results(results_folderpath)

    for r in results:
        measure=r.measure_result.measure
        # if r.measure_result.numpy.__class__.__name__ == tm.AnovaMeasure.__name__:
        #     print("anova")
        if r.measure_result.measure.__class__.__name__ == tm.NormalizedDistanceSameEquivariance:

            config.save_results(r,results_folderpath)
        else:
            pass
            #print(r.measure_result.numpy.__class__.__name__   )
            # print(r.measure_result.numpy.id())

