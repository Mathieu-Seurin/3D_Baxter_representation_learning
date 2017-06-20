
# coding: utf-8
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import os, os.path
import matplotlib

#DATASETS AVAILABLE:
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'
STATIC_BUTTON_SIMPLEST = 'staticButtonSimplest'

# 2 options of plotting:
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
GLOBAL_SCORE_LOG_FILE = 'globalScoreLog.csv'
MODELS_CONFIG_LOG_FILE  = 'modelsConfigLog.csv'
ALL_STATE_FILE = 'allStates.txt'
LAST_MODEL_FILE = 'lastModel.txt'
ALL_STATS_FILE ='allStats.csv'

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
    else:
        print "Unsupported dataset!"

"""
Use this function if rewards need to be visualized, use plot_3D otherwise
"""
def plotStates(mode, rewards, toplot, plot_path, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n', dataset=''): 
    # Plots states either learned or the ground truth
    # Useful documentation: https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
    # TODO: add vertical color bar for representing reward values  https://matplotlib.org/examples/api/colorbar_only.html
    reward_values = set(rewards)
    rewards_cardinal = len(reward_values)
    rewards = map(float, rewards)
    print'plotStates ',mode,' for rewards cardinal: ',rewards_cardinal,' (', reward_values,')'
    cmap = colors.ListedColormap(['green', 'blue', 'red'])  # TODO: adjust for different cardi$
    bounds=[-1,0,9,15] 
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")

    fig = plt.figure()
    if mode =='2D':
        ax = fig.add_subplot(111)#, projection = '2d')
        # colors_markers = [('r', 'o', -10, 0.5), ('b', '^', 0.5, 10)]
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], c=c, marker=m)
        ax.scatter(toplot[:,0], toplot[:,1], c=rewards, cmap=cmap, norm=norm, marker=".")#,fillstyle=None)
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=c, marker=m)
        ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=rewards, cmap=cmap, marker=".")#,fillstyle=None)
        ax.set_zlabel(axes_labels[2])
    else:
        print "only mode '2D' and '3D' plot supported"
        sys.exit(-1)

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    if 'GroundTruth' in plot_path:
        ax.set_title(title.replace('Learned Representations','Ground Truth')+dataset) 
    else:
        ax.set_title(title+dataset) 
    plt.show()
    plt.savefig(plot_path)
    print('\nSaved plot to '+plot_path)


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



#library_versions_tests()