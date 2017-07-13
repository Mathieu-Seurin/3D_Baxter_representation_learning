#!/usr/bin/python
#coding: utf-8

from __future__ import division

import numpy as np
import torch

import os
import subprocess

"""
This program adds synthetic extra negative reward to the images where the robot arm
is out of the field of view of the image (instead of the original 0s).
It is done manually by knowning the 3D coordinate space that belongs to the field of view (from ROS): see BOUND_INF and BOUND_SUP.
TODO: if impossible to do in ROS? apply same procedure to Babbling dataset of Leni (-> Is there a more precise way to get bounds
other than max and min (x, y, z) of arm position, respectively?)
"""

def is_in_bound(coordinate):
    BOUND_INF = [0.42,-0.09,-10] #-10 axis because we don't care about z axis at the moment
    BOUND_SUP = [0.74,0.59,10] #10 because we don't care about z axis at the moment

    for i,axis in enumerate(coordinate):
        if not(BOUND_INF[i] < axis < BOUND_SUP[i]):
            return False
    return True

database_folder =  'staticButtonSimplest/'
reward_file_name = 'recorded_button1_is_pressed.txt'

for record in os.listdir(database_folder):

    path_to_record = database_folder+record+'/'

    reward_file_whole_path = path_to_record+reward_file_name
    reward_file = open(reward_file_whole_path, 'r')

    state_file_whole_path = path_to_record+'recorded_robot_limb_left_endpoint_state.txt'
    state_file = open(state_file_whole_path, 'r')

    new_reward_file_content = ''

    len_reward_file = len(reward_file.readlines())
    len_state_file = len(state_file.readlines())

    reward_file.seek(0) #rewind file
    state_file.seek(0) #rewind file

    assert len_reward_file==len_state_file, "Need file to be the same length : Reward {} State {}".format(len_reward_file,len_state_file)

    for i in range(len_reward_file):

        line_rew_raw = reward_file.readline()
        line_rew = line_rew_raw.split(' ')

        line_state = state_file.readline().split(' ')

        if line_state[0] == '#':
            assert line_rew[0]=='#', "problem with comments"
            new_reward_file_content += line_rew_raw
        else:
            current_rew = int(line_rew[1])
            if current_rew == 0 :
                coordinate = [float(i) for i in line_state[1:]]

                if is_in_bound(coordinate):
                    #don't modify line
                    new_reward_file_content += line_rew_raw
                else:
                    new_reward_file_content += line_rew_raw[:-2]+'-1\n'
            else:
                new_reward_file_content += line_rew_raw


    reward_file.close()
    state_file.close()

    subprocess.call(["mv",reward_file_whole_path,reward_file_whole_path+'.old'])

    new_reward_file = open(reward_file_whole_path,'w')
    new_reward_file.write(new_reward_file_content)
    new_reward_file.close()
