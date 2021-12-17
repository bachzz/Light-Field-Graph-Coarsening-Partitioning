#!/bin/bash
#patch to be aware of "module" inside a job
#OAR -l /nodes=1/core=16,walltime = 48:0:0
#OAR -p mem_core > 2048
#OAR --array-param-file /$paraminputpath$/Param_build_graphs.txt
#OAR -O /$outputpath$/python_script_output.%jobid%.output
#OAR -E /$outputpath$/python_script_output.%jobid%.error

cd ~/virtualenv
source bin/activate

cd ~/$pythoncodepath$/
python -u ./Build_graph_PYTHON_v1.py $*
