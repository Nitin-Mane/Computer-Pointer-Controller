#!/usr/bin/env python3

#############################################################################################################
#
#                                  Gaze Estimator Script
#
#############################################################################################################

'''
Gaze Estimator Script 

Provides script to perform offline gaze estimation from eyetracking video frame and realtime operation. 
There are eye tracking classes and estimating the direction flow

'''

# Load system libraries
import os # load operating system library
import sys # load system library
import logging as log # load the logs library
# Load processing libraries
import numpy as np # load numerical analysis library
import cv2 # Load image processing library
# Load OpenVINO library
from openvino.inference_engine import IENetwork, IECore
from utils.ie_module import Module

class Gaze_Estimation(Module):
    """
    Gaze Estimator Class:
     For gaze estimation model it require three input blobs as follows
         1. right_eye_image
         2. head_pose_angles
         3. left_eye_image
    Input: 
          Load and configure inference plugins for the specified target devices and performs synchronous and asynchronous modes for the specified infer requests.
    Output: 
          Direction flow vectors
    """

    def __init__(self, model):
        
        # Initilization gaze estimation class
        # Setup inital stage
        super(Gaze_Estimation, self).__init__(model)
        
        assert len(model.inputs) == 3, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        # gaze estimation image input with the shape

        self.input_blob = [] 
        self.input_shape = []
        
        # Append the input and models shapes
        for inputs in model.inputs:
            self.input_blob.append(inputs)
            self.input_shape.append(model.inputs[inputs].shape)
        self.output_blob = next(iter(model.outputs))
    
    def enqueue(self, head_pose, right_eye, left_eye):
        # enqueue the gaze estimator input
        
        return super(Gaze_Estimation, self).enqueue({'left_eye_image': left_eye,
                                                    'right_eye_image': right_eye,
                                                    'head_pose_angles': head_pose})

    def start_async(self, headPosition, right_eye_image, left_eye_image):
        
        # Async process start
        head_pose = [headPosition.head_position_x, # x-axis
                    headPosition.head_position_y,  # y-axis
                    headPosition.head_position_z] # z-axis
        
        # set head post array
        head_pose = np.array([head_pose])
        head_pose = head_pose.flatten()
        
        # set the left eye information in the axis scale
        left_eye = cv2.resize(left_eye_image, (60, 60), interpolation = cv2.INTER_AREA)
        left_eye = np.moveaxis(left_eye, -1, 0)
        
        # set the right eye information in axis scale
        right_eye = cv2.resize(right_eye_image, (60, 60), interpolation = cv2.INTER_AREA)
        right_eye = np.moveaxis(right_eye, -1, 0)
        
        #enqueue the position and eye processed information
        self.enqueue(head_pose, right_eye, left_eye)
        
    def get_gazevector(self):
        
        # Getting the gaze vector information
        outputs = self.get_outputs()
        
        # the output generated a vectors 
        return (outputs)