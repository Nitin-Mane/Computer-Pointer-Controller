#########################################################################################################
#
#                                    Face Detection Script
#
#########################################################################################################
'''
Face Detection Script

This library helps for face dectection 

This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.
This has been provided just to give you an idea of how to structure your model class.
'''

# Load the system libraries 
import numpy as np 
from numpy import clip
from utils.ie_module import Module
from utils.helper import resize_input

'''
Face Dectector Class: 

This class define the class for the face detection library. 
'''



class Face_Detection(Module):
    '''
    This is the face detection class module
    It require the 
    the face input parameter are process for the finding the position and evaluating the outcome source file

    '''
    class Result:
        
        OUTPUT_SIZE_NUM = 7

        def __init__(self, outputfd):
            '''
            Initializing the face dectector function
            this will help to pass the image ID, labels, confidence range and the postion of the frame. 
            '''
            # taking the image id
            self.image_id = outputfd[0]
            # creating the label
            self.label = int(outputfd[1])
            # creating the confidence
            self.confidence = outputfd[2]
            # creating the position array
            self.position = np.array((outputfd[3], outputfd[4])) # (x, y)
            # resize the array outcome
            self.size = np.array((outputfd[5], outputfd[6])) # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            '''
            Rescale ROI: Specify a position constraint function inside the boundary of the image size.
            '''
            # position to set
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            # check the ROI scale factor size
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            '''
            Resize the ROI: Enable resizing of ROI object, specified as positon or size.
            '''
            # position outcome
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            '''
            Clip: Create the clip frames size of the min and max frame with position and size
            '''
            min = [0, 0]
            max = [width, height]
            self.position[:] = clip(self.position, min, max)
            self.size[:] = clip(self.size, min, max)
            

    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        '''
        Initilizing the model with the ROI scale factor

        Input: 
               Model 

        Outcome: 
                Shape
        '''
        super(Face_Detection, self).__init__(model)
        # finding the length of the model input and output
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        # Set the blob and the shape with respect to the model
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape
        # Set the outcome with the shape of the model size
        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE_NUM, \
            "Expected model output shape with %s outputs" % \
            (self.Result.OUTPUT_SIZE_NUM)
        # if the confidence threshold set to the 0.0 to 1.0 inbetween
        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        #checkthe threshold range
        self.confidence_threshold = confidence_threshold

        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        # set the ROI scale factor
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        # Image pre-processing technique for the reshaping the frame
        # Note - this is essential for the resizing the irregular and large video frame to fit model input
        assert len(frame.shape) == 4, "Frame shape should be in format as [1, c, h, w]"
        # Set the frame shape in the batch and c
        assert frame.shape[0] == 1 
        assert frame.shape[1] == 3
        # Resize input frame
        input = resize_input(frame, self.input_shape)

        return (input)

    def enqueue(self, input):
        #  insert function using list and super
        
        return (super(Face_Detection, self).enqueue({self.input_blob: input}))

    def start_async(self, frame):
        # Async technique for the pre-process frame
        # The async function is a coroutine return statements or the information which are essential
        input = self.preprocess(frame)
        # enqueue the input process
        self.enqueue(input)


    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]
        # set the frame width
        frame_width = frame.shape[-1]
        # set the frame height
        frame_height = frame.shape[-2]
        # process the result
        # set the result empty
        results = []
        # output progress in the result box format
        for outputfd in outputs[0][0]:
            # output model detection stage
            result = Face_Detection.Result(outputfd)
            # conditional statements of the confidence
            if result.confidence < self.confidence_threshold:
                break 
            # results are sorted by confidence decrease

            # resize the width results
            result.resize_roi(frame_width, frame_height)
            # rescale the ROI
            result.rescale_roi(self.roi_scale_factor)
            # Clip the frame width and height
            result.clip(frame_width, frame_height)
            # create the list format
            results.append(result)

        return (results)