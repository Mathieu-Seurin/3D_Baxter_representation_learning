# Baxter Robot Representation Learning 3D

This project is about a paper [1] written by Rico Jonschkowski and Oliver Brock. The goal is to learn a state representation based on images and robotics priors to make a network able to give high level information to another program which will make a robot learning tasks.

In this folder the network aims to learn a 3D representation of the robot hand position.

## DATA:

Place your data (from GDrive folder) in the main folder. The data folder should be named "simpleData3D".
If you use the 'colorful' dataset, beware that it will take 24GB of RAM.

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
IMPORTANT: In order to run it with a non random fixed test set of images,
call it with only one argument (the number of neigbors to generate for each
image in the test set and it will assess the test set of 50 images defined in Const.lua and Utils.py)

## RUNNING: python report_results.py
Will plot the current winning leaderboard of models' KNN MSE for each dataset trained.


## DEPENDENCIES


1. For nearest neighbors visualization:

* Scikit-learn:

Step 1: sudo apt-get update

Step 2: Install dependencies

sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base

Step 3:
pip install --user --install-option="--prefix=" scikit-learn

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
* Install torchnet-vision via https://github.com/Cadene/torchnet-vision  (requires luarocks install argcheck)

3. Pandas and Seaborn for Python results plotting and reporting:
pip install --user pandas
pip install --user seaborn (or  conda install seaborn
or
sudo apt-get pandas  ---see full scipy stack and add to readme

Mac install: cd /etc/   and $ Natalias-MacBook:etc natalia$ sudo nano tsocks.conf


4. When using ResNet, you
cd models and Download it remotely via:
wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
wget https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7
wget https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7  
# Beware: 101, 152 and 200 also exist!
and do:
require 'cunn'
require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md


## OPTIMIZATIONS
For cudnn memory/speed optimization options, see
https://github.com/soumith/cudnn.torch


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


2. SKLEARN AND SCIPY VERSION CONFLICTS: USE ONLY CONDA INSTALL SPECIFIC ONLINE COMMAND OR PIP INSTALL --user  *** WITHOUT SUDO ! ***


Do first (Requirements for scikit-learn) (Note: -U will give permission errors!):
```
pip install --user numpy
pip install --user scipy
```
If sklearn.neighbours import fails, remove  and install:
Either use conda (in which case all your installed packages would be in ~/miniconda/ or pip install --user don't mix the two. Removing either
```
rm -rf ~/.local/lib/python2.7/site-packages/sklearn or your ~/miniconda folder and reinstalling it cleanly should fix this.
sudo rm -rf scikit_learn-0.18.1.egg-info/
pip uninstall sklearn
```
and
```
1)  pip install --user scikit-learn
```
or
```
2) conda install -c anaconda scikit-learn=0.18.1
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

* If you run into this error :
```
tkinter.TclError: no display name and no $DISPLAY environment variable
```
while running ssh, Instead of ssh account@machine, do: ssh -X

* Running shell scripts:

The method has_command_finish_correctly for tracing errors better in console (break program and show error without continuing next program in the script) sometimes does not work. Solution: Run ./script.sh (without sh before). (If you get permission errors when running remotely via ssh, chmod u+x  script.sh)

* Lua/Torch allow only default boolean cmd line parameters of false. Add the flag to set them to True, leave them out to set them to False.

* Visualizing images: init.lua:389: module 'qt' not found:No LuaRocks module found for qt no field package.preload['qt']
Solution: for using qtlua, start torch in your terminal with:

$ qlua

instead of

$ th

* qlua: ./const.lua:149: module 'torchnet' not found:
	no field package.preload['torchnet']
luarocks install torchnet
luarocks install qtlua

* Q: bash script argument fails with error: invalid type for option -mcd (should be number)
A: the error disappears if using instead th instead of qlua. Some versions of Ubuntu 14.04 do not handle well qtlua, see versions.

* torchnet' not found
The issue can be related to qlua not being properly installed. Avoid installing from https://github.com/LuaDist/qtlua  and only INSTALL FROM: https://github.com/torch/qtlua
if luarocks install qtlua fails with
```
/tmp/luarocks_qtlua-scm-1-MmsVrW/qtlua/qtlua/qtluautils.cpp: In function ‘const char* pushnexttemplate(lua_State*, const char*)’:
/tmp/luarocks_qtlua-scm-1-MmsVrW/qtlua/qtlua/qtluautils.cpp:545:20: error: ‘LUA_PATHSEP’ was not declared in this scope
   while (*path == *LUA_PATHSEP) path++;  /* skip separators */
                    ^
/tmp/luarocks_qtlua-scm-1-MmsVrW/qtlua/qtlua/qtluautils.cpp:547:21: error: ‘LUA_PATHSEP’ was not declared in this scope
   l = strchr(path, *LUA_PATHSEP);  /* find next separator */
                     ^
qtlua/CMakeFiles/libqtlua.dir/build.make:72: recipe for target 'qtlua/CMakeFiles/libqtlua.dir/qtluautils.cpp.o' failed
make[2]: *** [qtlua/CMakeFiles/libqtlua.dir/qtluautils.cpp.o] Error 1
CMakeFiles/Makefile2:85: recipe for target 'qtlua/CMakeFiles/libqtlua.dir/all' failed
make[1]: *** [qtlua/CMakeFiles/libqtlua.dir/all] Error 2
Makefile:127: recipe for target 'all' failed
make: *** [all] Error 2
Error: Build error: Failed building.
```
Solution:  (as in https://github.com/torch/paths/issues/5):
Run it with Lua 5.1 instead of 5.2: http://lua-users.org/wiki/LuaRocksConfig
to switch among these:
TORCH_LUA_VERSION=LUA52 ./install.sh
and edit your ./bashrc PATH and LD_LIBRARY_PATH. Example of working ~/.bashrc file:
```
. /home/gpu_center/torch/install/bin/torch-activate
export http_proxy=your_proxy_url_and_port
export https_proxy=your_proxy_url_and_port
export CUDNN_PATH="/usr/local/cuda/lib64/libcudnn.so.5"
export CPATH=~/cuda:$CPATH
export LIBRARY_PATH=~/cuda:$LIBRARY_PATH
export LD_LIBRARY_PATH=~/cuda:$LD_LIBRARY_PATH
```

~/torch/install/bin/luarocks install qtlua

Chintala: I STRONGLY recommend that you guys use our packaged self-contained luajit+luarocks when using torch. With this repo: https://github.com/soumith/torch-distro
When there's system lua installed, things get messy with torch global install.
Or
1-paths source file should be revised to handle this issue. (first things first, might be the best solution)
2-configure luarocks, according to this page



If persists:
Option a) install CUDA 8 (although all this code is tested in CUDA 7.5), and compiling with GCC 5 instead.
Option b) Check your cuda version with nvcc --version, which nvcc, or with
cat /usr/local/cuda/version.txt     #Note: https://stackoverflow.com/questions/41714757/how-to-find-cuda-version-in-ubuntu
and adjust accordingly with the right version by upgrading your torch, cutorch (and likely nn) packages, they are probably old.
(as in https://github.com/torch/distro/issues/141 )
```
luarocks install torch  # you should be good to go, whenever a package is not recognized after being installed, upgrade torch first with this command and then install package.
luarocks install cutorch
luarocks install nn
luarocks install cunn
```
Option c) https://github.com/torch/cutorch/issues/175

~/qtlua/bld$ ../configure --help



Q: if some writing operation of model fails due to a previous run or first run, this may help:
mkdir Log   [if it does not exist]
rm lastModel.txt
rm -r preload_folder



## Observations on datasets

* Colorful75 converges in losses fast, as it has more images, around epoch 3-5 and therefore 5-10 epocs are enough, while for the rest of smaller #seqs (~50), the nr of epocs is 50.

But before changing this parameter while testing, make sure the error converges fast in the first iterations when running. Currently for Colorful75 it is at --------------Epoch : 17 ---------------
 [================================= 131/131 ============================>]  Tot: 4m36s | Step: 2s114ms   
Loss Temp	0.0033959985800469
Loss Prop	0.079365891051353
Loss Caus	0.14184848366082
Loss Rep	0.044838557231332
Loss Fix (BRING_CLOSER_REF_POINT) 	0.0047991881112797


## Visualizing Graphs
Use nngraph_visualization.lua
Pre-requirements:
sudo apt-get install graphviz -y
luarocks install nngraph


## Tests Notes:

* test_nn_graph.lua is an example of how nngraph should be done. Also a note: never use updateOutput and updateGradInput, Only use forward and backward. Basically, forward calls updateOutput + other stuff to retain the gradients etc. And backward calls updateGradInput + other stuff to retain gradients etc. In conclusion, it's better to call forward/backward because some models are doing more than just calling updateOutput etc.

## REFERENCES
[1] Learning state representations with robotic priors. Rico Jonschkowski, Oliver Brock, 2015.
http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf

[2]
```
mkdir ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6  ; scp -r gpu_center@uei18:~/baxter_representation_learning_3D/Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6/*.* ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6
;
: mkdir ./Log/modelY2017_D22_M06_H14M36S26_staticButtonSimplest_resnet_cont_MCD0_4_S0_6/NearestNeighbors

```
