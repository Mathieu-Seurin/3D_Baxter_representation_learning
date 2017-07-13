#!/usr/bin/python
#coding: utf-8

from __future__ import division
import numpy as np
import random
import sys
from Utils import parse_true_state_file, parse_repr_file, LAST_MODEL_FILE,get_data_folder_from_model_name,LEARNED_REPRESENTATIONS_FILE, PATH_TO_LINEAR_MODEL
from sklearn.metrics.pairwise import euclidean_distances
from random import randint

""" 
This File should be called after generateNNImages
"""

class Disentanglement:
    def __init__(self, data_folder, input_imgs, true_representations, learned_representations):
        self.images = input_imgs
        self.representations = true_representations
        self.learned_representations = learned_representations
        self.data_folder = data_folder
        self.n_batches = 1 # TODO randomize and repeat  len(images)  # we use all images and their representations available for computing the disentaglement metric
        self.path_to_linear_model = PATH_TO_LINEAR_MODEL+data_folder+'t7'
        print 'Disentanglement class initialized, creating/using linear model in ', self.path_to_linear_model
        
    def compute_disentanglement_metric_score(self):
        """
        Returns the classification accuracy of the disentanglement metric.
        As in Higgings 2017, good reconstructions are associated with entangled representations (lower disentanglement scores).
        Disentangled representations (high disentanglement scores) often result in blurry reconstructions.
        Uses the abs. linear difference between the inferred latent representations: a linear difference equation[ 
        or linear recurrence relation equates 0 to a polynomial that is linear in the various iterates of a variable—that is, 
        in the values of the elements of a sequence. The polynomial's linearity means that each of its terms has degree 0 or 1
        """
        DIMENSION_OUT = 3 # read from dataset #TODO generalize for DIMENSION_OUT
        n_repr_per_set = len(self.representations)/2
        total_z_diff = 0

        # 1. Choose a generative factor y ~ Unif[1...K], e.g. y = scale, shape, orientation. TODO: is there other factors than position of arm we could sample from? button/table position or color?
        # In our case, position of Baxter robot arm, (x, y, z), or one coordinate only?
        # 2. For a batch of L samples:
        avg_metric = 0
        for batch in range(self.n_batches):
            latent_v_set1 = []
            latent_v_set2 = []  
            datapoints = 0 # to be splitted into latent sets v1 and v2
            samples_per_batch = len(self.representations)  # L
            # a) Sample 2 sets of latent representations, latent_v_set1 and latent_v_set2 enforcing [v1,l]_k = [v2,l]_k 
            # if k=y (so that the value of factor k=y is kept fixed)
            y = randint(0, DIMENSION_OUT-1) # 1 coordinate ground truth or all?  We randomly choose one dimension index in [0,2]
            z_diff_batch = 0
            while datapoints <n_repr_per_set:
                if randint(0, 1):
                    latent_v_set1.append((self.images[datapoints], self.learned_representations[datapoints][y]))
                else:
                    latent_v_set2.append((self.images[datapoints], self.learned_representations[datapoints][y]))
                if latent_v_set1 == n_repr_per_set:
                    for (img, repres) in zip(self.images[datapoints:], self.learned_representations[datapoints:][y]):
                        latent_v_set2.append((img, repres)) 
                    return
                if latent_v_set2 == n_repr_per_set:
                    for (img, repres) in zip(self.images[datapoints:], self.learned_representations[datapoints:][y]):
                        latent_v_set1.append((img, repres)) 
                    return
                datapoints +=1
            print 'v1 and v2:\n',#latent_v_set1, latent_v_set2,'\n',
            print len(latent_v_set1), len(latent_v_set2)
            # b) Simuilate image x_1l,  ~ Sim(v_1l) and then infer z_1l = mu(x_1l) 
            # using the encoder q(z|x) ~ N (mu(x), sigma(x))-> i.e. our priors siamese model
            for (sample1, sample2) in zip(latent_v_set2, latent_v_set2):
                simulated_img1, encoded_img_z1 = sample1[0], sample1[1]
                simulated_img2, encoded_img_z2 = sample2[0], sample2[1]
                #print simulated_img1, encoded_img_z1 
                # encoded_img_z1 = representations[simulated_img1] # z_1l
                # encoded_img_z2 = representations[simulated_img2] # z_2l

                # c) compute the difference z_diff = |z_1l - z_2l|, the absolute linear difference between the inferred latent representations.
                z_diff_batch += np.absolute(np.subtract(encoded_img_z1, encoded_img_z2))
            avg_z_diff = z_diff_batch/float(samples_per_batch)
            total_z_diff += avg_z_diff
            print 'Disentanglement metric score for batch ',batch,': ', avg_z_diff

        # 3. Predict the factor y p(y|z_diff) and report the accuracy of this predictor as disentanglement metric score
        total_z_diff += total_z_diff/float(self.n_batches)
        factor_predictor_accuracy = 0
        for img in images:
            y_hat = self.predict_posterior_with_linear_model(img, y, total_z_diff)  # computes p(y|z_diff_batch)
            factor_predictor_accuracy += np.abs(np.substract(y_hat, learned_representations)) # predicting the real value of the factor (arm real position in our case)
        factor_predictor_accuracy /= len(images)
        print 'Disentanglement metric score for batch in dataset ',data_folder,': ',factor_predictor_accuracy
        return factor_predictor_accuracy

    def predict_posterior_with_linear_model(self, x, dim, given_condition_prior):
        """
        Linear classifier to learn the identity of the generative factor
        y is one of the generative factors (arm_pos_x, arm_pos_y, arm_pos_z)
         if multi_class is set to be “multinomial” the softmax function is used to find the predicted probability of each class.

        """
        # 1. Train model if model for this learned representation model does not exist
        # Save model 
        # or load trained model otherwise
        #y_hat = logistic_regr_classifier.predict()
        y_hat = x
        return y_hat


if len(sys.argv) >= 2:
    print "file used", sys.argv[1]
    path_to_model = sys.argv[1]
else:
    lastModelFile = open(LAST_MODEL_FILE)
    path_to_model = lastModelFile.readline()[:-1]

print 'Computing disentanglement_metric_score for model ',path_to_model

data_folder = get_data_folder_from_model_name(path_to_model)
file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE
images, representations = parse_repr_file(file_representation_string)

true_states = parse_true_state_file()
print representations
# convert learned states to numpy
learned_states = np.asarray([map(float,x) for x in representations])
print 'true and learnt states: \n',len(true_states), len(learned_states)
print learned_states

d = Disentanglement(data_folder, images, true_states, learned_states)
disentanglement_metric_score = d.compute_disentanglement_metric_score()

score_file = open(path_to_model+'/scoreNN.txt','a')
global_str = '\nDisentanglement Metric Score : '+str(disentanglement_metric_score)
score_file.write(global_str+'\n')
score_file.close()


