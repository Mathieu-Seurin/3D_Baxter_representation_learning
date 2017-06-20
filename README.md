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




## DEPENDENCIES
* torchnet and torchnet-vision
luarocks install torchnet-vision does not suffice:

1. luarocks install torchnet  
2. Install torchnet-vision via https://github.com/Cadene/torchnet-vision


Other:

1. For nearest neighbors visualization:

* Scikit-learn:

Step 1: sudo apt-get update

Step 2: Install dependencies

sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base

Step 3:
pip install --user --install-option="--prefix=" -U scikit-learn

* Tkinker: if you encounter "_tkinter.TclError: no display name and no $DISPLAY environment variable" while running
```
python generateNNImages.py
```

run instead 'ssh -X' instead of 'ssh'

or

```
import matplotlib
matplotlib.use('GTK')  # Or any other X11 back-end
```


## POTENTIAL ISSUES:

If using the dataset STATIC_BUTTON_SIMPLEST, the following error appears: `PANIC: unprotected error in call to Lua API (not enough memory)`

Do:
```
cd ~/torch
./clean.sh
TORCH_LUA_VERSION=LUA52 ./install.sh
And then everything should work
```

and after, reinstall torchnet and torchnet-vision as above indicated


## REFERENCES
[1] Learning state representations with robotic priors. Rico Jonschkowski, Oliver Brock, 2015.
http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf
