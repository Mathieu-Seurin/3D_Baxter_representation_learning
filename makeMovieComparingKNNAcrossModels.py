# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, produceRelevantImageStatesPlotMovie, get_movie_test_set_for_data_folder
from Utils import LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING, MOBILE_ROBOT, GIF_MOVIES_PATH, ALL_KNN_MOVIE_TEST_SETS, BENCHMARK_DATASETS, FOLDER_NAME_FOR_KNN_GIF_SEQ,PATH_TO_MOSAICS
#from Utils import STATIC_BUTTON_SIMPLEST, COMPLEX_DATA, COLORFUL75, COLORFUL, MOBILE_ROBOT
import numpy as np
import sys
import os.path
import subprocess
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ++++ This program assumes a folder containing all models where we have run  generate_neighbors_for_all_models_movie.sh


# def list_only_directories_in_path(given_path, recursively = False, containing_pattern_in_name =''):# = 'KNNGIFSeq'):
#     #NOT WORKING
#     dirs_paths = []
#     if  recursively:
#         for dir_, _, files in os.walk(given_path):
#             for fileName in files:
#                 if containing_pattern_in_name in fileName:
#                     relDir = os.path.relpath(dir_, given_path)
#                     relFile = os.path.join(relDir, fileName)
#                     dirs_paths.append(relFile) # relative path to each fiile
#                     print os.path.relpath((given_path))
#                     print relFile
#         return dirs_paths
#     else:
#         return [d for d in os.listdir(given_path) if os.path.isdir(os.path.join(given_path, d))]
#         # for d in os.listdir(given_path):
#         #     path_to_file = FOLDER_NAME_FOR_KNN_GIF_SEQ.replace('/', '') 
#         #     if containing_pattern_in_name in path_to_file and os.path.isdir(os.path.join(given_path, d)):
#         #         dirs_paths.append(path_to_file)
#         # return dirs_paths

# def list_only_files_in_path(given_path, containing_pattern_in_name= ''):
#NOT WORKING
#     list_of_files =[]
#     for f in os.listdir(given_path):
#         path_to_file = FOLDER_NAME_FOR_KNN_GIF_SEQ.replace('/', '') 
#         if containing_pattern_in_name in path_to_file and isfile(path_to_file):
#             list_of_files.append(path_to_file)
#     return list_of_files

def get_immediate_subdirectories_path(given_path, containing_pattern_in_name = ''):  # TODO add param for relative path vs just folder names
    return [name for name in os.listdir(given_path)
            if os.path.isdir(os.path.join(given_path, name)) and containing_pattern_in_name in os.path.join(given_path, name)]

def get_immediate_files_in_path(given_path, containing_pattern_in_name = ''):
    return [os.path.join(given_path, name) for name in os.listdir(given_path)
            if os.path.isfile(os.path.join(given_path, name)) and containing_pattern_in_name in os.path.join(given_path, name)]



def create_mosaic_img_and_save(input_reference_img_to_show_on_top, list_of_input_imgs, path_to_image_directory, output_file_name, top_title='', titles=[]):
    print "Creating mosaic from input reference image ", input_reference_img_to_show_on_top, '\nUsing images: ', list_of_input_imgs, 'saving it to ', path_to_image_directory

    if not os.path.exists(path_to_image_directory):
        os.mkdir(path_to_image_directory)

    numline = 4
    with_title = True
        
    fig = plt.figure()
    fig.set_size_inches(60,35)
    a=fig.add_subplot(numline+1,5,3)
    a.axis('off')
    # img = mpimg.imread(img_name)
    img = Image.open(input_reference_img_to_show_on_top)
    imgplot = plt.imshow(img)

    if with_title:
        a.set_title(top_title)


    for i in range(0, len(list_of_input_imgs)):
        a=fig.add_subplot(numline+1,5,6+i)
        img_name= list_of_input_imgs[i]
        # img = mpimg.imread(img_name)
        img = Image.open(img_name)
        imgplot = plt.imshow(img)

        if with_title:
            a.set_title(titles[i])
        a.axis('off')

    plt.tight_layout()
    output_file = path_to_image_directory+output_file_name

    plt.savefig(output_file, bbox_inches='tight')
    plt.close() # efficiency: avoids keeping all images into RAM

    print 'Created mosaic in ', output_file

def create_GIF_from_imgs_in_folder(folder_rel_path, output_file_name):

    print 'Made GIF from images in ', folder_rel_path, ' saved in ',output_file_name

FOLDER_CONTAINING_ALL_MODELS = './Log/ALL_MODELS_KNNS'
datasets = BENCHMARK_DATASETS


print"\n\n >> Running makeMovieFromComparingKNNAcrossModels.py... FOLDER_CONTAINING_ALL_MODELS: ", FOLDER_CONTAINING_ALL_MODELS#, '\nALL_KNN_MOVIE_TEST_SETS: ', ALL_KNN_MOVIE_TEST_SETS
model_name = ''
# from glob import glob
# paths = glob(FOLDER_CONTAINING_ALL_MODELS)
# print paths

for data_folder, test_set in zip(datasets, ALL_KNN_MOVIE_TEST_SETS):
    models_for_a_dataset = get_immediate_subdirectories_path(FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder)#list_only_directories_in_path(FOLDER_CONTAINING_ALL_MODELS)
    if len(models_for_a_dataset) ==4:
        GT_imgs = []
        AE_imgs = []
        supervised_imgs = []
        priors_imgs = []
        for model_folder in models_for_a_dataset:
            path_to_neighbors = FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder+'/'+model_folder+FOLDER_NAME_FOR_KNN_GIF_SEQ
            print 'Processing model ', model_folder, ' from dataset ', data_folder, '\n path_to_neighbors: ', path_to_neighbors
            if 'Supervised' in model_folder or 'supervised' in model_folder:
                supervised_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
                if len(supervised_imgs)>0:
                    if len(supervised_imgs)!=len(test_set):
                        print 'Sizes Model KNNs and Input KNNs: ', len(supervised_imgs), ' and ', len(test_set)
                        sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and supervised_imgs')
            elif 'AE' in model_folder:
                AE_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
                if len(AE_imgs) >0:
                    if len(AE_imgs) !=len(test_set):
                        print 'Sizes Model KNNs and Input KNNs:', len(AE_imgs), ' and ', len(test_set)
                        sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and AE_imgs')
            elif 'GT' in model_folder or 'ground_truth' in model_folder:
                GT_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
                if len(GT_imgs)>0:
                    if len(GT_imgs) !=len(test_set):
                        print 'Sizes Model KNNs and Input KNNs:', len(GT_imgs), ' and ', len(test_set)
                        sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and GT_imgs')
            else:
                if model_folder != '': #                print 'Processing Robotics priors model: ', path_to_neighbors
                    priors_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame') #list_only_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
                    if len(priors_imgs)>0:
                        if len(priors_imgs) !=len(test_set):
                            print 'Sizes Model KNNs and Input KNNs:', len(priors_imgs), ' and ', len(test_set)
                            sys.exit("The size of the image sets in each model's folder should coincide! It does not for data_folder "+data_folder+' and priors_imgs')
                else:
                    sys.exit('Missing model for robotic priors in folder: '+data_folder)


        # create mosaics
        if not os.path.exists(PATH_TO_MOSAICS):
            os.mkdir(PATH_TO_MOSAICS)
        if len(test_set)>0 and len(GT_imgs)>0 and len(AE_imgs)>0 and len(supervised_imgs)>0 and len(priors_imgs)>0 :
            print 'Stitching images into a mosaic for dataset ',data_folder, ' Using models: ', models_for_a_dataset
            index =0
            for input_img_test, gt, superv, ae, prior in zip(test_set, GT_imgs, supervised_imgs, AE_imgs, priors_imgs):
                path_to_mosaic_images = PATH_TO_MOSAICS+data_folder+'/'
                create_mosaic_img_and_save(input_img_test, [prior, ae, superv, gt], path_to_mosaic_images, 'mosaic_'+str(index)+'.jpg', top_title=get_data_folder_from_model_name(input_img_test), titles=['Robotic Priors', 'Denoising Auto-Encoder', 'Supervised (Robot Hand Position)', 'Ground Truth'])
                index +=1
                # TODO TEST ONLY
                if index==2:
                    break
            #TODO create_GIF_from_imgs_in_folder(path_to_mosaic_images, './DEMO_GIFs/'+data_folder+'_KNN.gif')
    else:
        print 'Missing models for dataset (must be 4 at least): ', data_folder, ' Skipping this dataset for now'
