# Baxter Robot Representation Learning 3D

This project is about a paper [1] written by Rico Jonschkowski and Oliver Brock. The goal is to learn a state representation based on images and robotics priors to make a network able to give high level information to another program which will make a robot learning tasks.

In this folder the network aims to learn a 3D representation of the robot hand position.

## DATA:

Place your data (from GDrive folder) in the main folder. The data folder should be named "simpleData3D".

## MODEL:

The function "save_model" in script.lua saves models for each test. The tests done are defined in the list "Tests_Todo". Each test trains the model with a particular combination of priors (the best one used now is the one with all the priors). Once you have saved a trained model, load_model.lua loads it using the variable "name"


## WORKFLOW PROCESS:

1. script.lua creates and optimize a model, saves it to a .t7 file and writes in LAST_MODEL_FILE ='lastModel.txt' the name and path of the model saved

2. imagesAndRepr.lua looks for the last model used (in LAST_MODEL_FILE), loads the model, calculate the representations for all images in DATA_FOLDER, and creates a saveImagesAndRepr.txt file (that contains a line per image path and its corresponding representations).

3. generateNNImages.py looks for the corresponding saveImagesAndRepr.txt and applies K Nearest Neigbors for visual evaluation purposes, i.e., to assess the quality of the state representations learnt. Run `python generateNN.py 50 `   to generate only 50 instead of all images.



Note: This repo is an extension of https://github.com/Mathieu-Seurin/baxter_representation_learning_1D to the 3D case: unsupervised learning of states for 3D representation learning



## RUNNING: script.lua or the shell pipeline scripts:

```
# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')
```
Example:

th script.lua -use_continuous -data_folder staticButtonSimplest

IMPORTANT:
-If you want to run grid search, ssh or batch processing pipelines such as learn_predict_plotStates.sh, make SKIP_RENDERING = true in Utils.py for the KNN images and plots to be saved properly and later copy them with scp for visualization, e.g. using [2].


## RUNNING: plotStates.py
To get a glimpse of how the ground truth states look like, run `plotStates.py` and set there the constant: plotGroundTruthStates = True

## RUNNING: generateNNImages.py
Example to run this program for a given trained model:

```
python generateNNImages.py 5 5 Log/modelY2017_D24_M06_H06M19S10_staticButtonSimplest_resnet_cont_MCD0_8_S0_4
```

## RUNNING: python report_results.py
Will plot the current winning leaderboard of models' KNN MSE for each dataset trained.


## DEPENDENCIES


1. For nearest neighbors visualization:

* Scikit-learn:

Step 1: sudo apt-get update

Step 2: Install dependencies

sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base

Step 3:
pip install --user --install-option="--prefix=" -U scikit-learn

* Tkinker: if you encounter `tkinter.TclError: no display name and no $DISPLAY environment variable` while running
```
python generateNNImages.py
```

run instead 'ssh -X' instead of 'ssh', or

```
import matplotlib
matplotlib.use('GTK')  # Or any other X11 back-end
```

2. torchnet and torchnet-vision
luarocks install torchnet-vision does not suffice:

* luarocks install torchnet  
* Install torchnet-vision via https://github.com/Cadene/torchnet-vision


3. Pandas for Python results plotting and reporting:
sudo apt-get pandas  ---see full scipy stack and add to readme

Mac install: cd /etc/   and $ Natalias-MacBook:etc natalia$ sudo nano tsocks.conf


4. When using ResNet, you
require 'cunn'
require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md



## POTENTIAL ISSUES:

1. If using the dataset STATIC_BUTTON_SIMPLEST, the following error appears: `PANIC: unprotected error in call to Lua API (not enough memory)`

Do:

```
cd ~/torch
./clean.sh
TORCH_LUA_VERSION=LUA52 ./install.sh
And then everything should work
```

and after, reinstall torchnet and torchnet-vision as above indicated


2. SKLEARN AND SCIPY VERSION CONFLICTS: USE ONLY CONDA INSTALL SPECIFIC ONLINE COMMAND OR PIP INSTALL -U _ WITHOUT SUDO.

If sklearn.neighbours import fails, remove  and install:
Either use conda (in which case all your installed packages would be in ~/miniconda/ or pip install --user don't mix the two. Removing either
```
rm -rf ~/.local/lib/python2.7/site-packages/sklearn or your ~/miniconda folder and reinstalling it cleanly should fix this.
sudo rm -rf scikit_learn-0.18.1.egg-info/
pip uninstall sklearn
```
and
```
1)  pip install -U scikit-learn
```
or
```
2) conda install -c anaconda scikit-learn=0.18.1
```

If needed, also do
```
pip install -U numpy
pip install -U scipy
```

3. Matplotlib: If plots are not showing properly reward colours, or datapoints too small, your version of matplotlib may be too old, it needs to be at least 2.0. Run test in Utils.py library_versions_tests().


4. 'luaJIT not enough memory' is a known problem of luaJIT that allow a maximum size for table of 2GB (in RAM, it's not related to GPU memory). There seems to be no way to avoid or change this limitation, and since the new database is bigger than that. 2 solutions:

- Either you delete some part of the database until you don't have the limitations (hotfix)
- Or you change the core of torch to Lua52 instead of luaJIT (what i did, but you need to re-install everything, torch cudnn etc...)

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch

# clean old torch installation
./clean.sh
# optional clean command (for older torch versions)
# curl -s https://raw.githubusercontent.com/torch/ezinstall/master/clean-old.sh | bash

# https://github.com/torch/distro : set env to use lua
TORCH_LUA_VERSION=LUA52 ./install.sh
```

Switching to lua52 worked

## KNOWN ERRORS:

* plot.figure() will fail when running the program ssh outside your proxy network (ENSTA with a Mac), but not when running the program on ssh within ensta with both machines being ubuntu:
```
Traceback (most recent call last):
  File "generateNNImages.py", line 128, in <module>
    fig = plt.figure()
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/pyplot.py", line 535, in figure
    **kwargs)
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5agg.py", line 44, in new_figure_manager
    return new_figure_manager_given_figure(num, thisFig)
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5agg.py", line 51, in new_figure_manager_given_figure
    canvas = FigureCanvasQTAgg(figure)
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5agg.py", line 242, in __init__
    super(FigureCanvasQTAgg, self).__init__(figure=figure)
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5agg.py", line 66, in __init__
    super(FigureCanvasQTAggBase, self).__init__(figure=figure)
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5.py", line 236, in __init__
    _create_qApp()
  File "/home/gpu_center/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_qt5.py", line 144, in _create_qApp
    raise RuntimeError('Invalid DISPLAY variable')
RuntimeError: Invalid DISPLAY variable
```

I have not found other solution than running within the proxy network (run locally within ensta or from another Ubuntu machine?)


## REFERENCES
[1] Learning state representations with robotic priors. Rico Jonschkowski, Oliver Brock, 2015.
http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf

[2]
```
mkdir ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6  ; scp -r gpu_center@uei18:~/baxter_representation_learning_3D/Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6/*.* ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6
;
: mkdir ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6/NearestNeighbors

```
