from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors

import sys

import numpy as np

if len(sys.argv) != 3:
        sys.exit("Give state file and reward file as arguments")

state_file=open(sys.argv[1])
reward_file=open(sys.argv[2])

states_l=[]
rewards_l=[]

if 'recorded_robot' in sys.argv[1] :
        for line in state_file:
                if line[0]!='#':
                        words=line.split(' ')
                        states_l.append([float(words[0]),float(words[1])])
        states=np.asarray(states_l)

        #The reward at step T correspond to the state at T+1, so we need to slide the state by one step backward
        #and remove the last reward to avoid 'out of bound' (Done below)
        toplot=states[1:]

else:
        last_rec = '' # to keep track of the current sequences we are parsing
        file_end = False
        while not file_end:
                line = state_file.readline()
                print "line",line 
                
                if (not line) or (line==' '):
                        file_end=True
                else:
                        words = line.split(' ')
                        num_record = words[0].split('/')[1][-3:]
                        #split the image name by dir, select the 2nd one (the sequence name)
                        # and take the last 3 char (the seq number)

                        if num_record == last_rec: #We are still in the same sequences
                                print "words",words[1:] 
                                states_l.append(np.array((map(float,words[1:-1]))))
                        else:
                                #if num_record != last_rec means => this is the first entry of the sequence
                                #Since we don't have the reward corresponding to the first state, you can ignore it.
                                last_rec = num_record
                                
        num_repr = len(states_l)
        size_repr = len(states_l[0])

        states = np.empty((num_repr,size_repr))
        for i in range(len(states_l)):
                #print states_l[i][1]
                states[i] = states_l[i]
        toplot=states

for line in reward_file:
	if line[0]!='#':
		words=line.split(' ')
		rewards_l.append(float(words[0]))

rewards=np.asarray(rewards_l)

if 'recorded_robot' in sys.argv[1]:
        #The reward at step T correspond to the state at T+1, so we need to slide the state by one step backward
        #and remove the last reward to avoid 'out of bound'
        rewards = rewards[:-1]
                

if states.ndim > 2:
        pca = PCA(n_components=2)
        pca.fit(states)
        toplot = pca.transform(states)

cmap = colors.ListedColormap(['blue', 'grey', 'red'])
bounds=[-1,0,9,15]
norm = colors.BoundaryNorm(bounds, cmap.N)
#cmap=plt.cm.plasma
# print toplot[0:10,0]
# print toplot[0:10,1]
# print rewards[0:10]

plt.scatter(toplot[:,0],toplot[:,1],c=rewards,cmap=cmap, norm=norm,marker="o")
plt.show()
