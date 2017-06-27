# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, plotStates
from Utils import BABBLING, MOBILE_ROBOT, SIMPLEDATA3D, PUSHING_BUTTON_AUGMENTED, STATIC_BUTTON_SIMPLEST, LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING
import numpy as np
import sys
import os.path
import subprocess
import unittest
test = unittest.TestCase('__init__')

# PLOTTING GROUND TRUTH OR LEARNED STATES
#####################
# True if we plot ground truth observed states, and false to plot the learned state representations
plotGroundTruthStates = False

library_versions_tests()
print"\n\n >> Running plotStates.py....plotGroundTruthStates: ",plotGroundTruthStates, " SKIP_RENDERING = ", SKIP_RENDERING

model_name = ''
if len(sys.argv) != 3:
    lastModelFile = open('lastModel.txt')
    path = lastModelFile.readline()[:-1]+'/'
    model_name = path.split('/')[1]
    # ONLY FOR FAST TESTING !!:   model_name = MOBILE_ROBOT#STATIC_BUTTON_SIMPLEST#'pushingButton3DAugmented' #TODO REMOVE-testing  model_name = MOBILE_ROBOT
    data_folder = get_data_folder_from_model_name(model_name)
    reward_file_str = 'allRewards_'+data_folder+'.txt'
    if plotGroundTruthStates:
        state_file_str = 'allStates_'+data_folder+'.txt'
        print "*********************\nPLOTTING GROUND TRUTH (OBSERVED) STATES for model: ", model_name#(Baxter left wrist position for 3D PUSHING_BUTTON_AUGMENTED dataset, or grid 2D position for MOBILE_ROBOT dataset)
        plot_path = path+'GroundTruthStatesPlot_'+model_name+'.png'
    else:
        state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
        print "*********************\nPLOTTING LEARNT STATES for model: ", model_name #(3D for Baxter PUSHING_BUTTON_AUGMENTED dataset, or 2D position for MOBILE_ROBOT dataset): ", state_file_str
        plot_path = path+'LearnedStatesPlot_'+model_name+'.png'
    lastModelFile.close()
else:
    state_file_str = sys.argv[1]
    reward_file_str = sys.argv[2]

    # if not os.path.isfile(state_file_str): # print('Calling subprocess create_plotStates_file_for_all_seq with ',data_folder)
    subprocess.call(['th','create_plotStates_file_for_all_seq.lua','-use_cuda','-use_continuous','-data_folder', data_folder])  # TODO: READ CMD LINE ARGS FROM FILE INSTEAD (and set accordingly here) TO NOT HAVING TO MODIFY INSTEAD train_predict_plotStates and the python files  
    # if not os.path.isfile(reward_file_str): #print('Calling subprocess create_all_reward with ',data_folder)
    subprocess.call(['th','create_all_reward.lua', '-use_cuda','-use_continuous','-data_folder', data_folder])


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
else: # general case
    print 'NAME_SAVE', state_file_str
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
        states[i] = np.array(states_l[i][1])


# Reading rewards
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
PLOT_DIMENSIONS = 3

if REPRESENTATIONS_DIMENSIONS >3:
    print "[Applying PCA to visualize the ",REPRESENTATIONS_DIMENSIONS,"D learnt representations space (PLOT_DIMENSIONS = ", PLOT_DIMENSIONS,")"
    pca = PCA(n_components=PLOT_DIMENSIONS) # default to 3
    pca.fit(states)
    toplot = pca.transform(states)
elif REPRESENTATIONS_DIMENSIONS ==2:
    PLOT_DIMENSIONS = 2 #    print "[PCA not applied since learnt representations' dimensions are not larger than 2]"
else:
    PLOT_DIMENSIONS = 3  # Default, if mobileData used, we plot just 2
#print "\n REPRESENTATIONS_DIMENSIONS =", REPRESENTATIONS_DIMENSIONS


if PLOT_DIMENSIONS == 2:
    plotStates('2D', rewards, toplot, plot_path, dataset=model_name)
elif PLOT_DIMENSIONS ==3:
    plotStates('3D', rewards, toplot, plot_path, dataset=model_name)
# elif PLOT_DIMENSIONS == 1:  #TODO  extend plotStates('1D') or allow cmap to run without gray -1 error
#     plt.scatter(toplot, rewards, c=rewards, cmap=cmap, norm=norm, marker="o")
else:
    print " PLOT_DIMENSIONS other than 2 or 3 not supported"





# def parse_arguments(): # TODO in future
#     skipRendering = False
#     import argparse

#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                        help='an integer for the accumulator')
#     parser.add_argument('--sum', dest='accumulate', action='store_const',
#                        const=sum, default=max,
#                        help='sum the integers (default: find the max)')

#     args = parser.parse_args()
#     print(args.accumulate(args.integers))
#     print "This is the name of the script: ", sys.argv[0]
#     print "Number of arguments: ", len(sys.argv)
#     print "The arguments are: " , str(sys.argv)
#     print "\n\n >> RUNNING plotStates.py  -skipRendering: ", skipRendering
#     print parser.parse_args()
#     return skipRendering
