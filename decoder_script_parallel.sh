#!/bin/bash
#patch to be aware of "module" inside a job
#OAR -l {mem_core>3000}/nodes=1/core=12,walltime=12:0:0
#OAR --array-param-file /$paraminputpath$/Param_decoder.txt
#OAR -O /$output_path$/decoder_output.%jobid%.output
#OAR -E /$output_path$/decoder_output.%jobid%.error

cd ~/virtualenv
source bin/activate

cd ~/$pythoncodepath$/
python -u ./Decoder_Mira.py $*
