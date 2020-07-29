#!/usr/bin/env python3

#######################################################################################################
#
#                                    Model Feeder Script
#
#######################################################################################################

# load the library
# load the system libary
import os.path as osp
import sys
import time

# load numerical operation library
import numpy as np
from math import cos, sin, pi 

# load the log librarys
import logging as log

# load OpenCV library
import cv2

# load the Argument Parser for user input
from argparse import ArgumentParser

# load the model and input feeder library (custom)
from utils.ie_module import Inference_Context
from utils.helper import cut_rois, resize_input
from .face_detection import Face_Detection
from .head_position_estimation import Head_Pose_Estimator
from .landmark_detection import Landmarks_Detection
from .gaze_Estimator import Gaze_Estimation
from .mouse_controller import Mouse_Controller_Pointer

# load the OpenVINO library
from openvino.inference_engine import IENetwork


class Process_On_Frame:
    
    '''
    Process the inference on the frames and generate the counter method.
    Queue size parameter is process to frames in a queue for Inference Engine
    1. Creating the Inference Engine module 
    2. Loading the device selection plugin
    3. load the face detection model on Inference 
    4. load the head position model on Inference
    5. load the land mark model on Inference 
    6. load the gaze estimator model on Inference
    7. Creating the time frame for loading model
    8. Configuring the model parameter inputs 
    9. Depolying the model parameter 
    10. Generating the logs file

    This is the basic process for the loading model and processing the outcome with the model inference.  
    '''
    
    QUEUE_SIZE_NUM = 1 # Queue size input
    
    def __init__(self, args):
        '''
        Initilizing the inference engine plugin with the device input parameter

        '''
        used_devices = set([args.d_fd, args.d_hp, args.d_lm, args.d_gm])
        
        # Creating a Inference Engine Context for the model
        self.context = Inference_Context()
        context = self.context

        # Loading OpenVino Plugin based on device selection and the library
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})
        # Loading the model performance for counter time
        log.info("Loading models")
        start_time = time.perf_counter()
        
        # Loading the face detection model on Inference Engine via argument
        face_detector_net = self.load_model(args.model_face_detection)
        
        # Loading Head position model on Inference Engine via argument
        head_position_net = self.load_model(args.model_head_position)

        # Loading the Landmark regressor'd model on Inference Engine via argument
        landmarks_net = self.load_model(args.model_landmark_regressor)

        # Load gaze estimation model on Intermediate Inference (IE) via argument 
        gaze_net = self.load_model(args.model_gaze)
        
        # Stop the time performance counter 
        stop_time = time.perf_counter()
        
        # Print the model time with execution on the stop to start time difference
        print("[INFO] Model Load Time: {}".format(stop_time - start_time))

        # Configure Face detector [detect threshold region, ROI Scale region]
        self.face_detector = Face_Detection(face_detector_net,
                                    confidence_threshold=args.t_fd,
                                    roi_scale_factor=args.exp_r_fd)
        
        # Configure the Head Pose Estimation network
        self.head_estimator = Head_Pose_Estimator(head_position_net)

        # Configure Landmark regressor network
        self.landmarks_detector = Landmarks_Detection(landmarks_net)
        
        # Configure Gaze Estimation network
        self.gaze_estimator = Gaze_Estimation(gaze_net)

        # Deploying Face detection device 
        self.face_detector.deploy(args.d_fd, context)
        
        # Deploying Head Position Detection device
        self.head_estimator.deploy(args.d_hp, context)

        # Deploying Landmark detector device
        self.landmarks_detector.deploy(args.d_lm, context)
        
        # Deploying Gaze Estimation device
        self.gaze_estimator.deploy(args.d_gm, context)

        # generating the log file
        log.info("Models are loaded")
    
    
    def load_model(self, model_path):
        """
        Initializing IENetwork(Inference Enginer) object from IR files:
        
        Args:
        Model path - This should contain both .xml and .bin file from the model path
        model path - < intel model pre-trained model >

        :return Instance of IENetwork class
        """
        # load the model path
        model_path = osp.abspath(model_path)
        
        # model path description
        model_description_path = model_path
        
        # model weights path in .bin
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        
        # generate the loading the model from the 
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
           
        # Load model on IE
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        
        # outcome of the model loaded
        return (model)


    
    def frame_pre_process(self, frame):
        """
        Pre-Process the input frame given to model

        Args:
        
        frame: 
              Input frame from video stream in the .mp4 and the cam

        Return:

        frame: 
               Pre-Processed frame [CHW]

        Image pre-processing technique
        """
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format proposed"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input process"
        # setup the frame in the original format
        
        #orig_image = frame.copy()
        original_image = frame.copy()
        
        # creating the frame transpose conversion
        frame = frame.transpose((2, 0, 1)) # Converting from HWC to CHW
        
        # creating the frame dimensions
        frame = np.expand_dims(frame, axis=0)
        
        # return the frames outcome
        return (frame)

    
    
    def face_detector_process(self, frame):
        """
        Face Detection Algorithm process

        Args:
              The Input Frame

        :return: 
              roi [xmin, xmax, ymin, ymax]
        """
        frame = self.frame_pre_process(frame)

        # Clear Face detector from previous frame
        self.face_detector.clear()

        # When we use async IE use buffer by using Queue
        self.face_detector.start_async(frame)

        # Predict and return ROI
        rois = self.face_detector.get_roi_proposals(frame)

        if self.QUEUE_SIZE_NUM < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE_NUM, len(rois)))
            rois = rois[:self.QUEUE_SIZE_NUM]
        
        self.rois = rois
        
        return (rois)

    
    
    def head_position_estimator_process(self, frame):
        
        """
        Head Position Estimator Process 

        Args:
                The Input Frame

        :return 
                headPoseAngles[angle_y_fc, angle_p_fc, angle_2=r_fc]
        """
        frame = self.frame_pre_process(frame)

        # Clean Head Position detection from previous frame
        self.head_estimator.clear()

        # Predict and return head position[Yaw, Pitch, Roll]
        self.head_estimator.start_async(frame, self.rois)
        headPoseAngles = self.head_estimator.get_headposition()

        return (headPoseAngles)

    
    
    def face_landmark_detector_process(self, frame):
        """
        Predict Face Landmark
        
        Args:
            The Input Frame

        :return:
            landmarks[left_eye, right_eye, nose_tip, left_lip_corner, right_lip_corner]
        """
        frame = self.frame_pre_process(frame)

        # Clean Landmark detection from previous frame
        self.landmarks_detector.clear()

        # Predict and return landmark detection[left_eye, right_eye, nose_tip, 
        # left_lip_corner, right_lip_corner]
        self.landmarks_detector.start_async(frame, self.rois)
        landmarks = self.landmarks_detector.get_landmarks()

        return (landmarks)

    
    
    def gaze_estimation_process(self, headPositon, right_eye, left_eye):
        """
        Predict Gaze estimation
        
        Args:
            The Input Frame

        :return:
            gaze_vector
        """

        # Clear gaze vector from the previous frame
        self.landmarks_detector.clear()
        
        # Get the gaze vector
        self.gaze_estimator.start_async(headPositon, right_eye, left_eye)
        gaze_vector = self.gaze_estimator.get_gazevector()
        return (gaze_vector)
    
    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'head_estimator': self.head_estimator.get_performance_stats(),
            'gaze_estimator': self.gaze_estimator.get_performance_stats(),
        }

        return (stats)