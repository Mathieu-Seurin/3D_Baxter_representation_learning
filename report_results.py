# coding: utf-8
import yaml
import pandas as pd
import numpy as np
import cv2
import math
import pandas
import sys, os, os.path

BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'


GLOBAL_SCORE_LOG_FILE = 'globalScoreLog.csv'
MODELS_CONFIG_LOG_FILE  = 'modelsConfigLog.csv'

DATA_FOLDER = MOBILE_ROBOT
#DATA_FOLDER = BABBLING

"""
This program uses baxter joint positions in cartesian space and translates
to discrete action space processable by representation_learning_3D program. For the wrist joint we need
the 3 axis rotations given by:
right_w0 right_w1 right_w2
Because each image in the recorded data produces many frame_ID values with joint values for that frame,
we keep only one set of joint values
The output files of the program look as follows:

OUTPUT_FILE (to be FILENAME_FOR_STATE in https://github.com/Mathieu-Seurin/baxter_representation_learning_3D)
         #time         x         y         z
0  289000578.0  0.816698  0.249241 -0.179920
1  487000576.0  0.757203  0.480915  0.400777
2  110000581.0  0.816698  0.249241 -0.179920
3  206000574.0  0.695541  0.432679  0.428267
4  787000579.0  0.816698  0.249241 -0.179920

second OUTPUT FILE (to be FILENAME_FOR_STATE_DELTAS)
         #time        dx        dy        dz
0  289000578.0  0.816698  0.249241 -0.179920
1  487000576.0 -0.059495  0.231674  0.580697
2  110000581.0  0.876193  0.017567 -0.760617
3  206000574.0 -0.180652  0.415112  1.188885
4  787000579.0  0.997350 -0.165870 -1.368805

OUTPUT_FILE_REWARD
         #time  value
0  289000578.0    0.0
1  487000576.0    0.0
2  110000581.0    0.0
3  206000574.0    0.0
4  787000579.0    0.0

"""
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
    ax.set_title(title+dataset)
    


################

####   MAIN program
################


if len(data_iteration_folders)==0:
    print OUTPUT_DIR," not found: No output data folder was created yet, run first real_data2simulated_data.py using the input directory ", INPUT_DIR
    sys.exit(-1)

records = pd.DataFrame(columns=('#time', 'x', 'y', 'z'))
records_rewards = pd.DataFrame(columns=('#time', 'value'))

positive_reward_ratio = 0

print "READING DATA FOLDER STRUCTURE DIRECTORIES: ",len(data_iteration_folders)
folder_pattern_index = 1
for folder in data_iteration_folders:
    record_folder = OUTPUT_DIR+ SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT.replace('X', str(folder_pattern_index))
    # reading each record_X folder
    if os.path.exists(record_folder):
        output_path = OUTPUT_DIR+ SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT.replace('X', str(folder_pattern_index))
        data_file = output_path +OUTPUT_FILE
        reward_file = output_path +OUTPUT_FILE_REWARD

        # adding each record's position data to a general overall dataframe
        data_df = pd.read_csv(data_file, sep='\t') #    df_deltas = pd.DataFrame(columns=('#time', 'dx', 'dy', 'dz'))
        rewards_df = pd.read_csv(reward_file, sep='\t')
        print data_df.head()
        print rewards_df.head()
        records = records.append(data_df)#, ignore_index= True)
        records_rewards = records_rewards.append(rewards_df)#, ignore_index=True)
        folder_pattern_index += 1

# all_data_df = pd.concat(records)
# all_rewards_df = pd.concat(records_rewards)
# Sorting frames, as they are not written in the original yml files in timely consecutive real order
records.sort_values(by='#time', inplace=True )
records_rewards.sort_values(by='#time', inplace=True )

print records.head()
print records_rewards.head()
print "Final data contains ", len(records), ' datapoints and ', len(records), ' rewards (', round(get_positive_reward_ratio(records_rewards), 3),'% positive rewards)'

# Plot all actions (position) data
print "PLOTTING ALL ACTUAL FILES WITHIN THE DIRECTORIES... Nr of records: ",len(data_iteration_folders), " in ",DATA_FOLDER
plot_states_and_rewards(records, records_rewards)



############ PLOT ALL EXPERIMENTS SCORES

# writing scores to global log for plotting and reporting
header = ['#MODEL', 'KNN_MSE']#MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA'] # TODO: JOIN
if os.path.isfile(GLOBAL_SCORE_LOG_FILE) and os.path.isfile(MODELS_CONFIG_LOG_FILE):
    global_scores_df = pd.DataFrame.read_csv(GLOBAL_SCORE_LOG_FILE, columns = header) #'MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD': MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD, 'CONTINUOUS_ACTION_SIGMA':CONTINUOUS_ACTION_SIGMA})
    models_header = ['#MODEL', 'MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA']
    models_df = pd.DataFrame.read_csv(MODELS_CONFIG_LOG_FILE, columns = header) 

    global_scores_df.sort_values(by='#MODEL', inplace=True )
    # Plot all actions (position) data
    print "PLOTTING ALL experiments scores for a varying number of MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD and  CONTINUOUS_ACTION_SIGMA  in ",DATA_FOLDER
    plot_states_and_rewards(records, records_rewards)
else:
    print 'Error: the following files must exist to plot MSE over configuration values: ',GLOBAL_SCORE_LOG_FILE, ' and ', MODELS_CONFIG_LOG_FILE
