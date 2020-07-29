#!/usr/bin/env python3

##########################################################################################################
#
#                                Head Positon Estimation Script
#
#########################################################################################################
'''
Head Position Estimation Script 

Head Pose Estimation research is focuses on the prediction of the pose of a human head in an image. 
More specifically it concerns the prediction of the Euler angles of a human head. 
The Euler angles consists of three values: yaw, pitch and roll.
(Doc)[https://towardsdatascience.com/head-pose-estimation-with-hopenet-5e62ace254d5#:~:text=As%20the%20name%20suggests%2C%20Head,%3A%20yaw%2C%20pitch%20and%20roll.]
'''
# Load libraries
import os # load the operating system library
import sys # load the system library
# load technical library
import cv2 # load the OpenCV library
from math import cos, sin, pi # load the math library for configuration
# load the log and arguments parser library
import logging as log
import argparse 
# load the external libray for pre-processing and model pipeline
from utils.ie_module import Module
from utils.helper import cut_rois, resize_input

class Head_Pose_Estimator(Module):
    '''
    Head Pose Estimator Class: 
    This help to detect the head position direction and movement of the person action.
    It provide a face dots containing a human face. Then the face process is expanded and transformed to a dots to suit the needs of later steps.
    '''
    class Result:
        
        def __init__(self,output):
            '''
            Initializing the head position in x,y,z-axis
            Interactive 3D Graphics
            [Info](https://www.youtube.com/watch?v=q0jgqeS_ACM&feature=youtu.be) 
            '''
            self.head_position_x = output["angle_y_fc"][0] #Yaw
            self.head_position_y = output["angle_p_fc"][0] #Pitch
            self.head_position_z = output["angle_r_fc"][0] #Roll

    def __init__(self, model):
        # initalizing the class
        super(Head_Pose_Estimator, self).__init__(model)
        
        # input 1 blob to the model length
        assert len(model.inputs) == 1, "Expected 1 input blob process"
        
        # output at the 3 blob
        assert len(model.outputs) == 3, "Expected 1 output blob process"
        
        # Process the blob data
        self.input_blob = next(iter(model.inputs)) # model input
        self.output_blob = next(iter(model.outputs)) # output blob
        
        # model resize formation
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        # pre-processing technique
        # the frame shape are reshaped with the model input parameter
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        
        # the input frames are shape and resize
        return (inputs)
    
    def enqueue(self, input):
        # enqueue: the head positions estimating the results in queue
        
        return super(Head_Pose_Estimator, self).enqueue({self.input_blob: input})
    
    def start_async(self, frame, rois):
        # Starting the async techniques for the frame
        
        inputs = self.preprocess(frame, rois)
        
        for input in inputs:
            # outcome in queue for the input frame for further process
            self.enqueue(input)

    def get_headposition(self):
        # Head position of the output
        
        outputs = self.get_outputs()
        
        # processing the head position results
        return (Head_Pose_Estimator.Result(outputs[0]))