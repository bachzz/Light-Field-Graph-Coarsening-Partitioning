#!/bin/bash
#patch to be aware of "module" inside a job
#OAR -l {mem_core>9000}/nodes=1/core=12,walltime=47:59:59
#OAR --array-param-file /$paraminputpath$/Param_coder.txt
#OAR -O /$outputpath$/python_script_parallel_output.%jobid%.output
#OAR -E /$outputpath$/python_script_parallel_output.%jobid%.error

cd ~/virtualenv
source bin/activate

cd ~/$pythoncodepath$/
python -u ./Coder_Mira.py $*
