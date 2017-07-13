#!/bin/bash

#./AE_learn_convert_plot.sh

cp hyperparamsSup.lua hyperparams.lua
./supervised_convert_plot.sh complexData

cp hyperparamsVanilla.lua hyperparams.lua
./learn_convert_plot.sh complexData

cp hyperparametersFix.lua hyperparams.lua
./learn_convert_plot.sh complexData

cp constFix.lua const.lua
./learn_convert_plot.sh complexData

