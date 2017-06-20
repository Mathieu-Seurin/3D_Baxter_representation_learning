# coding: utf-8

import pandas as pd
import numpy as np
import math
import sys, os, os.path

# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, plotStates
from Utils import BABBLING, MOBILE_ROBOT, SIMPLEDATA3D, PUSHING_BUTTON_AUGMENTED, STATIC_BUTTON_SIMPLEST, LEARNED_REPRESENTATIONS_FILE
from Utils import GLOBAL_SCORE_LOG_FILE, MODELS_CONFIG_LOG_FILE, ALL_STATS_FILE


################

####   MAIN program

############ PLOT ALL EXPERIMENTS SCORES
all_datasets = [BABBLING, MOBILE_ROBOT, SIMPLEDATA3D, PUSHING_BUTTON_AUGMENTED, STATIC_BUTTON_SIMPLEST]
header = ['Model','KNN_MSE','DATA_FOLDER','MODEL_ARCHITECTURE_FILE','MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA']

def plot_all_config_performance(df):
    # Plot all MSE_KNN scores for each experiment
    print "Reporting all experiments KNN_MSE scores for a varying number of MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD and  CONTINUOUS_ACTION_SIGMA "
    
def print_leader_board(df, datasets):
    for dataset in datasets:  
        sub_dataframe =  df[df['DATA_FOLDER'].notnull() & (df['DATA_FOLDER']== dataset)] #df.loc[df['DATA_FOLDER'] == dataset]  #df[['DATA_FOLDER'] == dataset]
        if len(sub_dataframe)>0:
            best_KNN_MSE = sub_dataframe.KNN_MSE.min()
            best_model_name = sub_dataframe.loc[sub_dataframe['KNN_MSE'] == best_KNN_MSE].Model.values[0]
            print "DATASET ", dataset, " Min KNN_MSE: ", best_KNN_MSE, ": ", best_model_name

# writing scores to global log for plotting and reporting
#header = ['Model', 'KNN_MSE']#MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA'] # TODO: JOIN
if os.path.isfile(GLOBAL_SCORE_LOG_FILE) and os.path.isfile(MODELS_CONFIG_LOG_FILE):
    mse_df = pd.read_csv(GLOBAL_SCORE_LOG_FILE)#, columns = header) #'MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD': MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD, 'CONTINUOUS_ACTION_SIGMA':CONTINUOUS_ACTION_SIGMA})
    models_df = pd.read_csv(MODELS_CONFIG_LOG_FILE)#, columns = header)   print mse_df.head()    print models_df.head()
    print "Initial log files contain ", len(mse_df), ' MSE datapoints and ', len(models_df), ' models experimented ' 
    all_scores_logs = mse_df.merge(models_df, on='Model', how='left') #all_scores_logs = mse_df.set_index('Model').join(models_df.set_index('Model'))
    final = all_scores_logs[header]
    # Sorting frames, as they are not written in the original yml files in timely consecutive real order
    all_scores_logs.sort_values(by='Model', inplace=True ) #print "All data joined: \n", final.head()
    print_leader_board(final, all_datasets)
    final.to_csv(ALL_STATS_FILE, header = header)

else:
    print 'Error: the following files must exist to plot MSE over configuration values: ',GLOBAL_SCORE_LOG_FILE, ' and ', MODELS_CONFIG_LOG_FILE
    sys.exit(-1)


