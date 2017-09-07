# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, produceRelevantImageStatesPlotMovie, get_movie_test_set_for_data_folder
from Utils import LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING, MOBILE_ROBOT, GIF_MOVIES_PATH, ALL_KNN_MOVIE_TEST_SETS, BENCHMARK_DATASETS, FOLDER_NAME_FOR_KNN_GIF_SEQ
#from Utils import STATIC_BUTTON_SIMPLEST, COMPLEX_DATA, COLORFUL75, COLORFUL, MOBILE_ROBOT
import numpy as np
import sys
import os.path
import subprocess
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
from os import listdir
from os.path import isfile, join


# ++++ This program assumes a folder containing all models where we have run  generate_neighbors_for_all_models_movie.sh


def create_mosaic_img_and_save(input_reference_img_to_show_on_top, list_of_input_imgs, path_to_image):
    print "output folder: ",path_to_image
    if not os.path.exists(path_to_image):
        os.mkdir(path_to_image)

    if use_test_set or nbr_images == -1:
        data = zip(images,indexes,distances,representations)
        if len(set(images).intersection(TEST_SET)) == 0:
            sys.exit('Error in generateNNImages.py: the TEST_SET for this dataset has not been properly defined in Utils.py. TEST_SET must contain a subset of the full set of images in DATA_FOLDER => which in this case is:',data_folder)
    else:
        print ('Using a random test set of images for KNN MSE evaluation...')
        data = random.sample(zip(images,indexes,distances,representations),nbr_images)


    # For each random selected images (or all images in nbr_images==-1), you take
    # the k-nearest neighbour in the REPRESENTATION SPACE (the first argv parameter)

    #As a quantitative measure, for the k nearest neighbour
    #you compute the distance between the state of the original image and
    #the images retrieved using knn on representation space

    total_error = 0 # to assess the quality of repr
    nb_tot_img = 0

    if nbr_neighbors<=5:
        numline = 1  # number of rows to show in the image of neigbours to be saved, for visibility
    elif nbr_neighbors<=10:
        numline = 2
    else:
        numline = 3

    # TODO: more efficient: for img_name in test_set.keys() revising data above: 
    # HOWEVER this needs to compute also in create_all_rewards and create_plotStates for the test set, separately and  an extra file. Is it fair comparison to test images for nearest neigbours that are seen during training?
    print 'nr images: ', nbr_neighbors, ' nbr of images: ', len(data), 'use_test_set ',use_test_set, ' of size: ', len(TEST_SET)#, TEST_SET
    for img_name,neigbour_indexes,dist,state in data:
        if use_test_set: #      print img_name   colorful75/record_073/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg
            if not(img_name in TEST_SET): 
                continue
        base_name= os.path.splitext(os.path.basename(img_name))[0]
        seq_name= img_name.split("/")[1]
        print('Processing ' + seq_name + "/" + base_name + ' ...'+base_name)
        fig = plt.figure()
        fig.set_size_inches(60,35)
        a=fig.add_subplot(numline+1,5,3)
        a.axis('off')
        # img = mpimg.imread(img_name)
        img = Image.open(img_name)
        imgplot = plt.imshow(img)
        state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"

        original_coord = true_states[img_name]

        if with_title:
            a.set_title(seq_name + "/" + base_name + ": \n" + state_str + '\n' + str(original_coord))

        for i in range(0,nbr_neighbors):
            a=fig.add_subplot(numline+1,5,6+i)
            img_name=images[neigbour_indexes[i+1]]
            # img = mpimg.imread(img_name)
            img = Image.open(img_name)
            imgplot = plt.imshow(img)

            base_name_n= os.path.splitext(os.path.basename(img_name))[0]
            seq_name_n= img_name.split("/")[1]

            dist_str = ' d=' + '{:.4f}'.format(dist[i+1])

            state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in representations[neigbour_indexes[i+1]]]) + "]"
            neighbour_coord = true_states[img_name]
            total_error += np.linalg.norm(neighbour_coord-original_coord)
            nb_tot_img += 1

            if with_title:
                a.set_title(seq_name_n + "/" + base_name_n + ": \n" + state_str + dist_str + '\n' + str(neighbour_coord))
            a.axis('off')


        plt.tight_layout()
        output_file = path_to_image + seq_name + "_" + base_name

        plt.savefig(output_file, bbox_inches='tight')
        plt.close() # efficiency: avoids keeping all images into RAM

    print 'Created mosaic in ',path_to_image

def create_GIF_from_imgs_in_folder(folder_rel_path, output_file_name):
    print 'GIF created in ',output_file_name



def list_only_directories_in_path(given_path, recursively = False, containing_pattern_in_name =''):# = 'KNNGIFSeq'):
    dirs_paths = []
    if  recursively:
        for dir_, _, files in os.walk(given_path):
            for fileName in files:
                if containing_pattern_in_name in fileName:
                    relDir = os.path.relpath(dir_, given_path)
                    relFile = os.path.join(relDir, fileName)
                    dirs_paths.append(relFile) # relative path to each fiile
                    print os.path.relpath((given_path))
                    print relFile
        return dirs_paths
    else:
        return [d for d in os.listdir(given_path) if os.path.isdir(os.path.join(given_path, d))]
        # for d in os.listdir(given_path):
        #     path_to_file = FOLDER_NAME_FOR_KNN_GIF_SEQ.replace('/', '') 
        #     if containing_pattern_in_name in path_to_file and os.path.isdir(os.path.join(given_path, d)):
        #         dirs_paths.append(path_to_file)
        # return dirs_paths

def list_only_files_in_path(given_path, containing_pattern_in_name= ''):
    list_of_files =[]
    for f in os.listdir(given_path):
        path_to_file = FOLDER_NAME_FOR_KNN_GIF_SEQ.replace('/', '') 
        if containing_pattern_in_name in path_to_file and isfile(path_to_file):
            list_of_files.append(path_to_file)
    return list_of_files

def get_immediate_subdirectories_path(given_path, containing_pattern_in_name = ''):
    return [name for name in os.listdir(given_path)
            if os.path.isdir(os.path.join(given_path, name))]

FOLDER_CONTAINING_ALL_MODELS = './Log/ALL_MODELS_KNNS'
datasets = BENCHMARK_DATASETS


print"\n\n >> Running makeMovieFromComparingKNNAcrossModels.py... FOLDER_CONTAINING_ALL_MODELS: ", FOLDER_CONTAINING_ALL_MODELS, '\nALL_KNN_MOVIE_TEST_SETS: ', ALL_KNN_MOVIE_TEST_SETS
print 'datasets: ', datasets
model_name = ''
# from glob import glob
# paths = glob(FOLDER_CONTAINING_ALL_MODELS)
# print paths

for data_folder, test_set in zip(datasets, ALL_KNN_MOVIE_TEST_SETS):
    models_for_a_dataset = get_immediate_subdirectories_path(FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder)#list_only_directories_in_path(FOLDER_CONTAINING_ALL_MODELS)
    print 'Stitching images into a mosaic for dataset ',data_folder
    GT_imgs = []
    AE_imgs = []
    supervised_imgs = []
    priors_imgs = []
    for model_folder in models_for_a_dataset:
        path_to_neighbors = FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder+'/'+model_folder+FOLDER_NAME_FOR_KNN_GIF_SEQ
        print 'Processing model ', model_folder, ' from dataset ',data_folder, ' path_to_neighbors: ', path_to_neighbors
        if 'Supervised' in model_folder:
            supervised_imgs = list_only_files_in_path(path_to_neighbors,'_frame')
            if len(supervised_imgs)>0:
                if len(supervised_imgs)!=len(test_set):
                    print 'Sizes: ', len(supervised_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and supervised_imgs')
        elif 'AE' in model_folder:
            AE_imgs = list_only_files_in_path(path_to_neighbors,'_frame')
            if len(AE_imgs) >0:
                if len(AE_imgs) !=len(test_set):
                    print 'Sizes: ', len(AE_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and AE_imgs')
        elif 'GT' in model_folder:
            GT_imgs = list_only_files_in_path(path_to_neighbors,'_frame')
            if len(GT_imgs)>0:
                if len(GT_imgs) !=len(test_set):
                    print 'Sizes: ', len(GT_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and GT_imgs')
        else:
            if model_folder != '':
                priors_imgs = list_only_files_in_path(path_to_neighbors,'_frame')
                if len(priors_imgs)>0:
                    if len(priors_imgs) !=len(test_set):
                        print 'Sizes: ', len(priors_imgs), ' and ', len(test_set)
                        sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and priors_imgs')
            else:
                sys.exit('Missing model for robotic priors in folder: '+data_folder)

    if len(test_set)>0 and len(GT_imgs)>0 and len(AE_imgs)>0 and len(supervised_imgs)>0 and len(priors_imgs)>0 :
        index =1
        for gt, superv, ae, prior in zip(GT, supervised, AEs, GTs):
            path_to_image = data_folder+'/KNN_Comparison_Dataset_'+data_folder+'.gif'
            joint_img = create_mosaic_img_and_save([gt, superv, ae, prior], path_to_image)
            index +=1
        create_GIF_from_imgs_in_folder(folder, output_file_name+data_folder)
    else:
        print 'Missing models for dataset: ', data_folder
