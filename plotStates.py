from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
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


def plot_3D(x =[1,2,3,4,5,6,7,8,9,10], y =[5,6,2,3,13,4,1,2,4,8], z =[2,3,3,3,5,7,9,11,9,10], axes_labels = ['U','V','W'], title='Learned representations-rewards distribution\n', dataset=''):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')  # 'r' : red

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title+dataset)
    # plt.savefig(plot_path)
    # plt.show()

model_name = ''
if len(sys.argv) != 3:
    lastModelFile = open('lastModel.txt')
    path = lastModelFile.readline()[:-1]+'/'
    state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
    lastModelFile.close()
    reward_file_str = 'allRewards.txt'
    model_name = path.split('/')[1]
    #print 'Plotting rewards for model: ', model_name
else:
    state_file_str = sys.argv[1]
    reward_file_str = sys.argv[2]

state_file=open(state_file_str)
reward_file=open(reward_file_str)

states_l=[]
rewards_l=[]

if 'recorded_robot' in state_file_str :
    print 'Plotting ', MOBILE_ROBOT,' observed states and rewards in ',state_file_str
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
    toplot=states

else:
    print 'Plotting learnt states and rewards in ',state_file_str, ' for model: ', model_name
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
REPRESENTATIONS_DIMENSIONS = len(states[0])

if REPRESENTATIONS_DIMENSIONS >2: #.ndim >2:
    print "[Applying PCA to visualize the learnt representations space, with PLOT_DIMENSIONS = ", PLOT_DIMENSIONS
    pca = PCA(n_components=PLOT_DIMENSIONS)
    pca.fit(states)
    toplot = pca.transform(states)

    #cmap=plt.cm.plasma
    # print toplot[0:10,0]
    # print toplot[0:10,1]
    # print rewards[0:10]
else:
    PLOT_DIMENSIONS = 2
    print "[PCA not applied since learnt representations' dimensions are not larger than 2]"
print "Visualizing states with #REPRESENTATIONS_DIMENSIONS =", REPRESENTATIONS_DIMENSIONS, ' in ',PLOT_DIMENSIONS,'D'

cmap = colors.ListedColormap(['blue', 'grey', 'red'])  # TODO: adjust for different cardinal of reward types according to dataset
bounds=[-1,0,9,15] #TODO: parametrize according to the dataset
norm = colors.BoundaryNorm(bounds, cmap.N)
if PLOT_DIMENSIONS == 2:
    plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
elif PLOT_DIMENSIONS ==3: 
    plot_3D(toplot[:,0], toplot[:,1], toplot[:,2], dataset=model_name)
elif PLOT_DIMENSIONS == 1:
    plt.scatter(toplot[:,0], rewards, c=rewards, cmap=cmap, norm=norm,marker="o")
else:
    print " PLOT_DIMENSIONS undefined"

if PLOT_DIMENSIONS == 3: 
    d1_mean, d2_mean, d3_mean  = toplot[:,0].mean(), toplot[:,1].mean(), toplot[:,2].mean()
    d1_std, d2_std, d3_std  = toplot[:,0].std(), toplot[:,1].std(), toplot[:,2].std()
    print 'Means of the dimensions: ', d1_mean, d2_mean, d3_mean 
    print "Std devs of the dimensions: ",d1_std, d2_std, d3_std 
else:
    d1_mean = d2_mean = d3_mean = d1_std = d2_std = d3_std  = ''

plot_path = path+'ActionsSparsityPlot_'+model_name+'.png'
print('Saving plot to '+plot_path)
plt.savefig(plot_path)
plt.show()
