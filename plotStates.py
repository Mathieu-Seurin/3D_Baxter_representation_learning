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

def plot_3D(x =[1,2,3,4,5,6,7,8,9,10], y =[5,6,2,3,13,4,1,2,4,8], z =[2,3,3,3,5,7,9,11,9,10], axes_labels = ['U','V','W'], title='Learned representations-rewards distribution\n', dataset=''):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')  # 'r' : red

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title+dataset) # TODO: add to plot title the dataset name with getDatasetNameFromModelName()
    # plt.savefig(plot_path)
    # plt.show()

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
        plot_path = path+'GroundTruthStatesSparsityPlot_'+model_name+'.png'
    else:
        state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
        print "*********************\nPLOTTING LEARNT STATES (3D for Baxter PUSHING_BUTTON_AUGMENTED dataset, or 2D position for MOBILE_ROBOT dataset): ", state_file_str
        plot_path = path+'LearnedStatesSparsityPlot_'+model_name+'.png'
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

total_rewards = 0#sum(1 for line in reward_file)
total_states = 0 #sum(1 for line in state_file)        
states_l=[]
rewards_l=[]

if 'recorded_robot' in state_file_str :
    print 'Plotting ', MOBILE_ROBOT,' observed states and rewards in ',state_file_str
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
    #toplot=states
else:
    # if 'pushing_object' in state_file_str : #if  DATA_FOLDER == BABBLING:
    #     print 'Plotting ', BABBLING,' learnt states '
    # elif 'robot_limb_left_endpoint' in state_file_str : #if DATA_FOLDER == SIMPLEDATA3D 
    #     print 'Plotting ', SIMPLEDATA3D,' learnt states '
    # else:
    #     print 'ERROR: Unsupported dataset, pattern not found in ', state_file_str,'. Make sure you run first script.lua and imagesAndRepr.lua with the right DATA_FOLDER setting'
    #     sys.exit(-1)
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

cmap = colors.ListedColormap(['blue', 'grey', 'red'])  # TODO: adjust for different cardinal of reward types according to dataset
bounds=[-1,0,9,15] #TODO: parametrize according to the dataset?
norm = colors.BoundaryNorm(bounds, cmap.N)

if PLOT_DIMENSIONS == 2:
    plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS ==3: 
    plot_3D(toplot[:,0], toplot[:,1], toplot[:,2], dataset=model_name)     #plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS == 1:
    plt.scatter(toplot[:,0], rewards, c=rewards, cmap=cmap, norm=norm,marker="o")
else:
    print " PLOT_DIMENSIONS undefined"

print('\nSaved plot to '+plot_path)
plt.savefig(plot_path)
plt.show()



# if PLOT_DIMENSIONS == 3: 
#     d1_mean, d2_mean, d3_mean  = toplot[:,0].mean(), toplot[:,1].mean(), toplot[:,2].mean()
#     d1_std, d2_std, d3_std  = toplot[:,0].std(), toplot[:,1].std(), toplot[:,2].std()
#     #print 'Means of the dimensions: ', d1_mean, d2_mean, d3_mean  #    print "Std devs of the dimensions: ",d1_std, d2_std, d3_std 
# else:
#     d1_mean = d2_mean = d3_mean = d1_std = d2_std = d3_std  = ''
