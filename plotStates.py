from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import numpy as np

#DATASETS AVAILABLE:
#TODO: REMOVE TO AVOID CONFLICT WITH const.lua values
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'

LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
#DATA_FOLDER = MOBILE_ROBOT
#DATA_FOLDER = BABBLING

PLOT_DIMENSIONS = 3

model_name = ''
if len(sys.argv) != 3:
    lastModelFile = open('lastModel.txt')
    path = lastModelFile.readline()[:-1]+'/'
    state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
    lastModelFile.close()
    reward_file_str = 'allRewards.txt'
    model_name = path.split('/')[1]
    print 'Plotting rewards for model: ', model_name
else:
    state_file_str = sys.argv[1]
    reward_file_str = sys.argv[2]

state_file=open(state_file_str)
reward_file=open(reward_file_str)

states_l=[]
rewards_l=[]

if 'recorded_robot' in state_file_str :
    print 'Plotting ', MOBILE_ROBOT,' observed states in ',state_file_str
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
    toplot=states

else:
    print 'Plotting learnt states in ',state_file_str
    # if 'pushing_object' in state_file_str : #if  DATA_FOLDER == BABBLING:
    #     print 'Plotting ', BABBLING,' learnt states '
    # elif 'robot_limb_left_endpoint' in state_file_str : #if DATA_FOLDER == SIMPLEDATA3D 
    #     print 'Plotting ', SIMPLEDATA3D,' learnt states '
    # else:
    #     print 'ERROR: Unsupported dataset, pattern not found in ', state_file_str,'. Make sure you run first script.lua and imagesAndRepr.lua with the right DATA_FOLDER setting'
    #     sys.exit(-1)

    for line in state_file:
        if line[0]!='#':
            # Saving each image file and its learned representations
            words=line.split(' ')
            states_l.append((words[0],list(map(float,words[1:-1]))))

    states_l.sort(key= lambda x : x[0])

    states = np.zeros((len(states_l), len(states_l[0][1])))
    for i in range(len(states_l)):
        #print states_l[i][1]
        states[i] = np.array(states_l[i][1])

# getting rewards 
for line in reward_file:
    if line[0]!='#':
        words=line.split(' ')
        rewards_l.append(float(words[0]))

rewards=np.asarray(rewards_l)
toplot=states

if states.ndim >2:
    print "[Applying PCA to visualize the learnt representations space, with PLOT_DIMENSIONS = ", PLOT_DIMENSIONS
    pca = PCA(n_components=PLOT_DIMENSIONS)
    pca.fit(states)
    toplot = pca.transform(states)

    #cmap=plt.cm.plasma
    # print toplot[0:10,0]
    # print toplot[0:10,1]
    # print rewards[0:10]
else:
    print "[PCA not applied since learnt representations' dimensions are not larger than 2]"
    PLOT_DIMENSIONS = 2

cmap = colors.ListedColormap(['blue', 'grey', 'red'])  # TODO: adjust for different cardinal of reward types according to dataset
bounds=[-1,0,9,15] #TODO: parametrize according to the dataset
norm = colors.BoundaryNorm(bounds, cmap.N)
if PLOT_DIMENSIONS == 2:
    plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS ==3: 
    print('Plotting 3D...')
    plt.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS == 1:
    plt.scatter(toplot[:,0], rewards, c=rewards, cmap=cmap, norm=norm,marker="o")
else:
    print " PLOT_DIMENSIONS undefined"

plot_path = path+'learnedStatesPlot_'+model_name+'.png'
print('Saving plot to '+plot_path)
plt.savefig(plot_path)
plt.show()
