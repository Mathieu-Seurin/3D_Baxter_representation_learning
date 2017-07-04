
# coding: utf-8
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import os, os.path
import matplotlib
import seaborn as sns

"""
Documentation for colorblind-supported plots: #http://seaborn.pydata.org/introduction.html
"""

SKIP_RENDERING = True  # Make False only when running remotely via ssh for plots and KNN figures to be saved

#DATASETS AVAILABLE:
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'
STATIC_BUTTON_SIMPLEST = 'staticButtonSimplest'
COMPLEXDATA = 'complexData'

# 2 options of plotting:
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
GLOBAL_SCORE_LOG_FILE = 'globalScoreLog.csv'
MODELS_CONFIG_LOG_FILE  = 'modelsConfigLog.csv'
ALL_STATE_FILE = 'allStates.txt'
LAST_MODEL_FILE = 'lastModel.txt'
ALL_STATS_FILE ='allStats.csv'
LUA2PYTHON_COMMUNICATION_FILE = 'lua2pythonCommunication.txt' # not used yet, TODO


def library_versions_tests():
    if not matplotlib.__version__.startswith('2.'):
        print "Using a too old matplotlib version (can be critical for properly plotting reward colours, otherwise the colors are difficult to see), to update, you need to do it via Anaconda: "
        print "Min version required is 2.0.0. Current version: ", matplotlib.__version__
        print "Option 1) (Preferred)\n - pip install --upgrade matplotlib"
        print "2) To install anaconda (WARNING: can make sklearn PCA not work by installing a second version of numpy): \n -wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh  \n -bash Anaconda2-4.4.0-Linux-x86_64.sh  \n -Restart terminal \n -conda update matplotlib"
        sys.exit(-1)

    numpy_versions_installed = np.__path__
    #print "numpy_versions_installed: ", numpy_versions_installed
    if len(numpy_versions_installed)>1:
        print "Probably you have installed numpy with and without Anaconda, so there is a conflict because two numpy versions can be used."
        print "Remove non-Anaconda numpy:\n 1) pip uninstall numpy \n and if needed, install 2.1) pip install -U numpy  \n "
        print "2.2) If 1 does not work: last version in: \n -https://anaconda.org/anaconda/numpy"

def get_data_folder_from_model_name(model_name):
    if BABBLING in model_name:
        return BABBLING
    elif MOBILE_ROBOT in model_name:
        return MOBILE_ROBOT
    elif SIMPLEDATA3D in model_name:
        return SIMPLEDATA3D
    elif PUSHING_BUTTON_AUGMENTED in model_name:
        return PUSHING_BUTTON_AUGMENTED
    elif STATIC_BUTTON_SIMPLEST in model_name:
        return STATIC_BUTTON_SIMPLEST
    elif COMPLEXDATA in model_name:
        return COMPLEXDATA
    else:
        print "Unsupported dataset!"

"""
Use this function if rewards need to be visualized, use plot_3D otherwise
"""
def plotStates(mode, rewards, toplot, plot_path, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n', dataset=''):
    # Plots states either learned or the ground truth
    # Useful documentation: https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
    # Against colourblindness: https://chrisalbon.com/python/seaborn_color_palettes.html
    # TODO: add vertical color bar for representing reward values  https://matplotlib.org/examples/api/colorbar_only.html
    reward_values = set(rewards)
    rewards_cardinal = len(reward_values)
    rewards = map(float, rewards)
    print'plotStates ',mode,' for rewards cardinal: ',rewards_cardinal,' (', reward_values,')'
    cmap = colors.ListedColormap(['gray', 'blue', 'red'])     # print "cmap: ",type(cmap)

    # custom Red Gray Blue colormap
    colours = [(0.3,0.3,0.3), (0.0,0.0,1.0), (1,0,0)]
    n_bins = 100
    cmap_name = 'rgrayb'
    cm = LinearSegmentedColormap.from_list(cmap_name, colours, n_bins)

    colorblind_palette = sns.color_palette("colorblind", rewards_cardinal)  # 3 is the number of different colours to use
    #print(type(colorblind_palette))
    #cubehelix_pal_cmap = sns.cubehelix_palette(as_cmap=True)
    #print(type(cubehelix_pal_cmap))

    #sns.palplot(sns.color_palette("colorblind", 10))
    # sns.color_palette()
    #sns.set_palette('colorblind')

    colorblind_cmap  = ListedColormap(colorblind_palette)
    colormap = cmap
    bounds=[-1,0,9,15]
    norm = colors.BoundaryNorm(bounds, colormap.N)
    # TODO: for some reason, no matther if I use cmap=cmap or make cmap=colorblind_palette work, it prints just 2 colors too similar for a colorblind person

    fig = plt.figure()
    if mode =='2D':
        ax = fig.add_subplot(111)
        # colors_markers = [('r', 'o', -10, 0.5), ('b', '^', 0.5, 10)]
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], c=c, marker=m)
        ax.scatter(toplot[:,0], toplot[:,1], c=rewards, cmap=cmap, norm=norm, marker=".")#,fillstyle=None)
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=c, marker=m)
        cax = ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=rewards, cmap=cm, marker=".")#,fillstyle=None)
        ax.set_zlabel(axes_labels[2])
    else:
        sys.exit("only mode '2D' and '3D' plot supported")

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    if 'GroundTruth' in plot_path:
        ax.set_title(title.replace('Learned Representations','Ground Truth')+dataset)
    else:
        ax.set_title(title+dataset)

    # adding color bar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar
    plt.savefig(plot_path)
    #plt.colorbar()  #TODO WHY IT DOES NOT SHOW AND SHOWS A PALETTE INSTEAD?
    if not SKIP_RENDERING:  # IMPORTANT TO SAVE BEFORE SHOWING SO THAT IMAGES DO NOT BECOME BLANK!
        plt.show()
    print('\nSaved plot to '+plot_path)
    plt.show()


"""
Use this function if rewards DO NOT need to be visualized, use plotStates otherwise
"""
def plot_3D(x =[1,2,3,4,5,6,7,8,9,10], y =[5,6,2,3,13,4,1,2,4,8], z =[2,3,3,3,5,7,9,11,9,10], axes_labels = ['U','V','W'], title='Learned representations-rewards distribution\n', dataset=''):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')  # 'r' : red

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title+dataset)


def file2dict(file): # DO SAME FUNCTIONS IN LUA and call at the end of set_hyperparams() method SKIP_VISUALIZATIOn, USE_CUDA and all the other params to used them in the subprocess subroutine.
    d = {}
    with open(file) as f:
        for line in f:
            if line[0]!='#':
               key_and_values = line.split()
               key, values = key_and_values[0], key_and_values[1:]
               d[key] = map(float, values)
    return d


# TODO : extend for other datasets for comparison
IMG_TEST_SET = {
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00065.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00078.jpg',

'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',

'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00063.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',

'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00074.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',

'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',

'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00070.jpg',

'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',

'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00053.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00083.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',

'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',

'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00059.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00089.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',

'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00064.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',

'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00068.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg'}


#library_versions_tests()
