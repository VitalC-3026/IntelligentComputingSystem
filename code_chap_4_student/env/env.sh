#!/bin/bash
export AICSE_MODELS_MODEL_HOME=$PWD/../data/models
export AICSE_MODELS_DATA_HOME=$PWD/../data/data
export NEUWARE=$PWD/neuware
export NEUWARE_HOME=$PWD/neuware
export TENSORFLOW_MODELS_DATA_HOME=$AICSE_MODELS_DATA_HOME
export PATH=$PATH:$NEUWARE/bin
export PATH=$PATH:/usr/local/neuware/bin
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NEUWARE/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/neuware/lib64
source /etc/profile
