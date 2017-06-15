from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import os, os.path
import unittest 
test = unittest.TestCase('__init__')

#DATASETS AVAILABLE:
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'

# 2 options of plotting:
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
#ALL_GROUND_TRUTH_STATES_FILE = 

plotGroundTruthStates = False 
# true if we plot ground truth observed states, and false to plot the learned state representations

def get_data_folder_from_model_name(model_name):
    if BABBLING in model_name:
        return BABBLING
    elif MOBILE_ROBOT in model_name:
        return MOBILE_ROBOT
    elif SIMPLEDATA3D in model_name:
        return SIMPLEDATA3D
    elif PUSHING_BUTTON_AUGMENTED in model_name:
        return PUSHING_BUTTON_AUGMENTED
    else:
        print "Unsupported dataset!"

def plotStates(mode, rewards, toplot, plot_path, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n', dataset=''): 
    reward_values = set(rewards)
    rewards_cardinal = len(reward_values)
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

model_name = ''
if len(sys.argv) != 3:
    lastModelFile = open('lastModel.txt')
    path = lastModelFile.readline()[:-1]+'/'
    model_name = path.split('/')[1]
    data_folder = get_data_folder_from_model_name(model_name)
    reward_file_str = 'allRewards_'+data_folder+'.txt'
    if plotGroundTruthStates:
        state_file_str = 'allStates_'+data_folder+'.txt'
        print "*********************\nPLOTTING GROUND TRUTH (OBSERVED) STATES (Baxter left wrist position for 3D PUSHING_BUTTON_AUGMENTED dataset, or grid 2D position for MOBILE_ROBOT dataset): ", state_file_str#, ' for model: ', model_name
        plot_path = path+'GroundTruthStatesPlot_'+model_name+'.png'
    else:
        state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
        print "*********************\nPLOTTING LEARNT STATES (3D for Baxter PUSHING_BUTTON_AUGMENTED dataset, or 2D position for MOBILE_ROBOT dataset): ", state_file_str
        plot_path = path+'LearnedStatesPlot_'+model_name+'.png'
    lastModelFile.close()
    #print 'Plotting rewards for model: ', model_name
else:
    state_file_str = sys.argv[1]
    reward_file_str = sys.argv[2]

if not os.path.isfile(state_file_str):
    #state_file=open(state_file_str)
    print 'ERROR: states file does not exist: ', state_file_str,'. Make sure you run first script.lua and imagesAndRepr.lua with the right DATA_FOLDER setting, as well as create_all_reward.lua and create_plotStates_file_for_all_seq.lua'
    sys.exit(-1)
if not os.path.isfile(reward_file_str):
    reward_file=open(reward_file_str)
    print 'ERROR: rewards file does not exist: ', reward_file_str,'. Make sure you run first script.lua and imagesAndRepr.lua with the right DATA_FOLDER setting, as well as create_all_reward.lua and create_plotStates_file_for_all_seq.lua'
    sys.exit(-1)

total_rewards = 0 
total_states = 0 
states_l=[]
rewards_l=[]

if 'recorded_robot' in state_file_str :
    print 'Plotting ', MOBILE_ROBOT,' observed states and rewards in ',state_file_str
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
else:
    with open(state_file_str) as f:
        for line in f:
            if line[0]!='#':
                # Saving each image file and its learned representations
                words=line.split(' ')
                states_l.append((words[0], list(map(float,words[1:-1]))))
                total_states += 1

    states_l.sort(key= lambda x : x[0])
    states = np.zeros((len(states_l), len(states_l[0][1])))

    for i in range(len(states_l)):
        #print states_l[i][1]
        states[i] = np.array(states_l[i][1])

# getting rewards 
with open(reward_file_str) as f:
    for line in f:
        if line[0]!='#':
            words=line.split(' ')
            rewards_l.append(words[0])
            total_rewards+= 1
rewards=rewards_l
toplot=states

print "Ploting total states and total rewards: ",total_states, " ", total_rewards," in files: ",state_file_str," and ", reward_file_str
test.assertEqual(total_rewards, total_states, "Datapoints size discordance! Length of rewards and state files should be equal, and it is "+str(len(rewards))+" and "+str(len(toplot))+" Run first create_all_reward.lua and create_plotStates_file_for_all_seq.lua")

REPRESENTATIONS_DIMENSIONS = len(states[0])

if REPRESENTATIONS_DIMENSIONS >3: 
    print "[Applying PCA to visualize the ",REPRESENTATIONS_DIMENSIONS,"D learnt representations space (PLOT_DIMENSIONS = ", PLOT_DIMENSIONS,")"
    pca = PCA(n_components=PLOT_DIMENSIONS) # default to 3
    pca.fit(states)
    toplot = pca.transform(states)

    #cmap=plt.cm.plasma
    # print toplot[0:10,0]
    # print toplot[0:10,1]
    # print rewards[0:10]
elif REPRESENTATIONS_DIMENSIONS ==2:
    PLOT_DIMENSIONS = 2 #    print "[PCA not applied since learnt representations' dimensions are not larger than 2]"
else:
    PLOT_DIMENSIONS = 3  # Default, if mobileData used, we plot just 2
print "\nVisualizing states with #REPRESENTATIONS_DIMENSIONS =", REPRESENTATIONS_DIMENSIONS, ' in ',PLOT_DIMENSIONS,'D'


if PLOT_DIMENSIONS == 2:
    plotStates('2D', rewards, toplot, plot_path, dataset=model_name)     
    # fig = plt.figure()
    # ax = fig.add_subplot(111)#, projection = '2d')
    # for c, m, zlow, zhigh in [('r', 'o', -10, 0.4), ('b', '^', 0.5, 10)]:
    #     ax.scatter(toplot[:,0],toplot[:,1], c=c, marker=m)
    #plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS ==3:
    plotStates('3D', rewards, toplot, plot_path, dataset=model_name)    
# elif PLOT_DIMENSIONS == 1:  #TODO  extend plotStates('1D') or allow cmap to run without gray -1 error
#     plt.scatter(toplot, rewards, c=rewards, cmap=cmap, norm=norm, marker="o")
else:
    print " PLOT_DIMENSIONS undefined"
