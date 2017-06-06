from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import numpy as np

#DATASETS AVAILABLE:
#TODO: REMOVE TO AVOID CONFLICT WITH const.lua values
# BABBLING = 'babbling'
# MOBILE_ROBOT = 'mobileRobot'
# SIMPLEDATA3D = 'simpleData3D'

LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
#DATA_FOLDER = MOBILE_ROBOT
#DATA_FOLDER = BABBLING

PLOT_DIMENSIONS = 2

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

print 'Using saved states file: ',state_file_str

if 'recorded_robot' in state_file_str :
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
    toplot=states

else:
    #if DATA_FOLDER == SIMPLEDATA3D or DATA_FOLDER == BABBLING:
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

    # else:
    #     print('Unsupported dataset')
    #     sys.exit(-1)

# getting rewards 
for line in reward_file:
    if line[0]!='#':
        words=line.split(' ')
        rewards_l.append(float(words[0]))

rewards=np.asarray(rewards_l)
toplot=states
#rewards=np.asarray(rewards_l)


if states.ndim > 2 and PLOT_DIMENSIONS == 2:
    pca = PCA(n_components=2)
    pca.fit(states)
    toplot = pca.transform(states)

    cmap = colors.ListedColormap(['blue', 'grey', 'red'])  # TODO: adjust for different cardinal of reward types according to dataset
    bounds=[-1,0,9,15]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #cmap=plt.cm.plasma
    # print toplot[0:10,0]
    # print toplot[0:10,1]
    # print rewards[0:10]

    plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")

# elif PLOT_DIMENSIONS ==3: #TODO
#     print('Plotting 3D...')
#     plt.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=rewards,cmap=cmap, norm=norm,marker="o")

plot_path = path+'learnedStatesPlot_'+model_name+'.png'
print('Saving plot to '+plot_path)
plt.savefig(plot_path)
plt.show()
