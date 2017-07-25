#!/bin/bash

if [ "$1" != "" ]; then
    DATA_FOLDER=$1
else
    echo No data folder given as parameter
    exit
fi

# Priors without any addition
cp hyperparamsVanilla.lua hyperparams.lua
./learn_convert_plot.sh $DATA_FOLDER

# Prior with fix point above button
cp hyperparamsFix.lua hyperparams.lua
cp constDef.lua const.lua
./learn_convert_plot.sh $DATA_FOLDER

# Prior with fix point selected randomly
cp constFix.lua const.lua
./learn_convert_plot.sh $DATA_FOLDER

# Auto-encoder
./AE_learn_convert_plot.sh $DATA_FOLDER

#Supervised Learning
cp hyperparamsSup.lua hyperparams.lua
./supervised_convert_plot.sh $DATA_FOLDER
