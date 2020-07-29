#!/usr/bin/env python3

#####################################################################################################
#
#                                      Landmark Detection Script
#
######################################################################################################

'''
Landmark Detection Script: 
Facial landmarks are used to localize and represent salient regions of the face
This are following 
1. Eyes
2. Nose 
3. Mouth 

Detecting facial landmark frames by frames into a two step process:

Step #1: Localize the face in the image frame and preprocessing.
Step #2: Detect the key facial structures on the face ROI frames.

[docs](https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e)
'''
# load the library
import numpy as np # load numerical operation library
# load external libary for model pipeline and pre-processing frames.
from utils.helper import cut_rois, resize_input
from utils.ie_module import Module

class Landmarks_Detection(Module):
    '''
    Land Marks Detection Class
    this class represent the processing of the land mark operation.
    It will generate the land mark map of the face and it can be process for the detection process. 

    the main process is to pre-process the frames images and extract the ROI regions and add facial structures nodes points.
    This is will happen with help of the following inputs 
    1. Left Eye
    2. Right Eye
    3. Nose Tip 
    4. Left lips corner
    5. Right lips corner 

    the points are mapped based on the region detected and passsing the nearby area pixels
    '''
    POINTS_NUMBER = 5

    class Result:
        # Result class 
        
        # Initialize the class
        def __init__(self, outputs):
            
            # output set
            self.points = outputs
            
            #lambda: A lambda function is a small anonymous function
            lm = lambda i: self[i]
            
            # left eyes pointer
            self.left_eye = lm(0)
            
            # right eyes pointer
            self.right_eye = lm(1)
            
            # nose tip pointer
            self.nose_tip = lm(2)
            
            # left lips corner pointer
            self.left_lip_corner = lm(3)
            
            # right lip corner pointer
            self.right_lip_corner = lm(4)
        
        def __getitem__(self, idx):
            
            # Get the item list index
            return self.points[idx]

        def get_array(self):
            
            # get the array points
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        
        # Initialize the landmark model
        super(Landmarks_Detection, self).__init__(model)

        self.update = False
        
        # update the input and out
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        
        # blob process
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        
        # pre-processing technique
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        
        # input provides the frame with ROI region
        inputs = cut_rois(frame, rois)
        
        # resizing the frame and processing the shape input it
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        
        # enqueue the landmark detector modules
        return super(Landmarks_Detection, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        
        # Starting the async technique process
        inputs = self.preprocess(frame, rois)
        
        # When the input are process it get in queue
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        
        # Outcome of landmarks
        outputs = self.get_outputs()
        
        # output processed
        results = [Landmarks_Detection.Result(out[self.output_blob].reshape((-1, 2))) \
                      for out in outputs]
        
        # Outcome
        return (results)