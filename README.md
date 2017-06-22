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



## REFERENCES
[1] Learning state representations with robotic priors. Rico Jonschkowski, Oliver Brock, 2015.
http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf
