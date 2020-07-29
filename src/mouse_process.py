#!/usr/bin/env python3

############################################################################################################
#
#                              Mouse Process Script
#
############################################################################################################


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
from .model_feeder import Process_On_Frame

class Mouse_Controller:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, args):
        self.frame_processor = Process_On_Frame(args)
        self.display = True
        self.print_perf_stats = args.perf_stats
        # Setup the instruction arguments input
        self.fd_out = args.o_fd # Face detection
        self.hp_out = args.o_hp # Head position
        self.lm_out = args.o_lm # Land mark detection
        self.gm_out = args.o_gm # Gaze detection
        self.mc_out = args.o_mc # Mouse counter

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1
        self.right_eye_coords = None
        self.left_eye_coords = None 
        
        # Most controller
        self.mc = Mouse_Controller_Pointer('medium','fast')
        
        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1
    
    
    def update_fps(self):
        """
        Calculate FPS time
        """
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        return (self.fps)
        
    
    def draw_detection_roi(self, frame, roi):
        """
        Draw Face detection bounding Box

        Args:
        frame: 
              The Input Frame
        roi: 
            [xmin, xmax, ymin, ymax]
        """
        for i in range(len(roi)):
            # Draw face ROI border
            cv2.rectangle(frame,
                        tuple(roi[i].position), tuple(roi[i].position + roi[i].size),
                        (0, 220, 0), 2)

    
    def createEyeBoundingBox(self, point1, point2, scale=1.8):
        """
        Create a Eye bounding box using Two points that we got from headposition model

        Args:
        point1: 
                First Point coordinate
        point2: 
                Second Point coordinate
        """

        # Normalize the two points
        size  = cv2.norm(np.float32(point1) - point2)
        width = int(scale * size)
        height = width
        
        # Find x, y mid point
        midpoint_x = (point1[0] + point2[0]) / 2
        midpoint_y = (point1[1] + point2[1]) / 2

        # Calculate eye x, y point
        startX = midpoint_x - (width / 2)
        startY = midpoint_y - (height / 2)
        return [int(startX), int(startY), int(width), int(height)]

    
    def landmarkPostProcessing(self, frame, landmarks, roi, org_frame):
        """
        Calculate right eye bounding box and left eye bounding box by using
        landmark keypoints

        Args:
        frame: 
               Frame to resize/crop
        landmark: 
               Keypoints
        ROI: 
               Detection output of Facial detection model
        org_frame: 
               Orginal frame

        return:

               list of left and right bounding box
        """
        # setup the face bounding box width and height
        faceBoundingBoxWidth = roi[0].size[0]
        faceBoundingBoxHeight = roi[0].size[1]
        # creating the landmark pointers from the cropping the eyes
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]
        # Land marks pre-process using computer vision algorithm
        '''
        [link](https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/)
        '''
        faceLandmarks = []
        # setup the edge detection ROI of the left eye
        left_eye_x = (landmarks.left_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        left_eye_y = (landmarks.left_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        # adding the face land marks points x and y axis
        faceLandmarks.append([left_eye_x, left_eye_y])
        
        # setup the edge detection ROI of the right eye
        right_eye_x = (landmarks.right_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        right_eye_y = (landmarks.right_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        # adding the face land marks points of x and y axis
        faceLandmarks.append([right_eye_x, right_eye_y])
        
        # setup the corner egde detection ROI of nose tip
        nose_tip_x = (landmarks.nose_tip[0] * faceBoundingBoxWidth + roi[0].position[0])
        nose_tip_y = (landmarks.nose_tip[1] * faceBoundingBoxHeight + roi[0].position[1])
        # adding the face land marks points of x and y axes
        faceLandmarks.append([nose_tip_x, nose_tip_y])
        # setup the corner edge detection ROI of left lips
        left_lip_corner_x = (landmarks.left_lip_corner[0] * faceBoundingBoxWidth + roi[0].position[0])
        left_lip_corner_y = (landmarks.left_lip_corner[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        # facial landmarks corner axis
        faceLandmarks.append([left_lip_corner_x, left_lip_corner_y])
        # Left Eye box setup
        leftEyeBox = self.createEyeBoundingBox(faceLandmarks[0], 
                                    faceLandmarks[1],
                                    1.8)

        # Right Eye box Setup
        RightEyeBox = self.createEyeBoundingBox(faceLandmarks[2], 
                                    faceLandmarks[3],
                                    1.8)
        
        # To crop image using the eye bounding boxes
        # img[y:y+h, x:x+w]
        # setup the outcome
        leftEyeBox_img = org_frame[leftEyeBox[1] : leftEyeBox[1] + leftEyeBox[3], 
                             leftEyeBox[0] : leftEyeBox[0] + leftEyeBox[2]]

        # processing the right eye box arrays and processing to the image
        RightEyeBox_img = org_frame[RightEyeBox[1] : RightEyeBox[1] + RightEyeBox[3], 
                             RightEyeBox[0] : RightEyeBox[0] + RightEyeBox[2]]
        # Outcome
        return (RightEyeBox_img, leftEyeBox_img)

    
    def draw_final_result(self, frame, roi, headAngle, landmarks, gaze_vector):
        """
        Draw the final output on frame including facial detection input, 
        face landmarks, head angles and gaze vector
        """

        faceBoundingBoxWidth = roi[0].size[0]
        faceBoundingBoxHeight = roi[0].size[1]

        if self.fd_out:     
            # Draw Face detection bounding Box
            for i in range(len(roi)):
                # Draw face ROI border
                cv2.rectangle(frame,
                            tuple(roi[i].position), tuple(roi[i].position + roi[i].size),
                            (0, 0, 255), 4)

        # Draw headPoseAxes
        # Here head_position_x --> angle_y_fc  # Yaw
        #      head_position_y --> angle_p_fc  # Pitch
        #      head_position_z --> angle_r_fc  # Roll
        yaw = headAngle.head_position_x
        pitch = headAngle.head_position_y
        roll = headAngle.head_position_z

        sinY = sin(yaw * pi / 180.0)
        sinP = sin(pitch * pi / 180.0)
        sinR = sin(roll * pi / 180.0)

        cosY = cos(yaw * pi / 180.0)
        cosP = cos(pitch * pi / 180.0)
        cosR = cos(roll * pi / 180.0)
        
        axisLength = 0.4 * faceBoundingBoxWidth
        xCenter = int(roi[0].position[0] + faceBoundingBoxWidth / 2)
        yCenter = int(roi[0].position[1] + faceBoundingBoxHeight / 2)

        if self.hp_out:   
            #center to right
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * (cosR * cosY + sinY * sinP * sinR))),
                            ((yCenter) + int (axisLength * cosP * sinR))),
                            (0, 0, 255), thickness=2)
            #center to top
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * (cosR * sinY * sinP + cosY * sinR))),
                            ((yCenter) - int (axisLength * cosP * cosR))),
                            (0, 255, 0), thickness=2)
            
            #Center to forward
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * sinY * cosP)),
                            ((yCenter) + int (axisLength * sinP))),
                            (255, 0, 0), thickness=2)
        
        # Draw landmark 
        keypoints = [landmarks.left_eye,
                landmarks.right_eye,
                landmarks.nose_tip,
                landmarks.left_lip_corner,
                landmarks.right_lip_corner]
        
        if self.lm_out:
            for point in keypoints:
                center = roi[0].position + roi[0].size * point
                cv2.circle(frame, tuple(center.astype(int)), 2, (255, 255, 0), 4)
            
        # Draw Gaz vector with final frame
        left_eye_x = (landmarks.left_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        left_eye_y = (landmarks.left_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        right_eye_x = (landmarks.right_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        right_eye_y = (landmarks.right_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        nose_tip_x = (landmarks.nose_tip[0] * faceBoundingBoxWidth + roi[0].position[0])
        nose_tip_y = (landmarks.nose_tip[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        left_lip_corner_x = (landmarks.left_lip_corner[0] * faceBoundingBoxWidth + roi[0].position[0])
        left_lip_corner_y = (landmarks.left_lip_corner[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        leftEyeMidpoint_start = int(((left_eye_x + right_eye_x)) / 2)
        leftEyeMidpoint_end = int(((left_eye_y + right_eye_y)) / 2)
        rightEyeMidpoint_start = int((nose_tip_x + left_lip_corner_x) / 2)
        rightEyeMidpoint_End = int((nose_tip_y + left_lip_corner_y) / 2)
        
        # Gaze out
        arrowLength = 0.4 * faceBoundingBoxWidth
        gaze = gaze_vector[0]
        gazeArrow_x = int((gaze[0]) * arrowLength)
        gazeArrow_y = int(-(gaze[1]) * arrowLength)

        if self.gm_out:
            cv2.arrowedLine(frame, 
                            (leftEyeMidpoint_start, leftEyeMidpoint_end), 
                            ((leftEyeMidpoint_start + gazeArrow_x), 
                            leftEyeMidpoint_end + (gazeArrow_y)),
                            (0, 255, 0), 3)

            cv2.arrowedLine(frame, 
                            (rightEyeMidpoint_start, rightEyeMidpoint_End), 
                            ((rightEyeMidpoint_start + gazeArrow_x), 
                            rightEyeMidpoint_End + (gazeArrow_y)),
                            (0, 255, 0), 3)
        
        
        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())
            
    def get_mouse_point(self, headPosition, gaze_vector):
        yaw = headPosition.head_position_x
        pitch = headPosition.head_position_y
        roll = headPosition.head_position_z
        
        sinR = sin(roll * pi / 180.0)
        cosR = cos(roll * pi / 180.0)

        gaze_vector = gaze_vector[0]
        mouse_x = gaze_vector[0] * cosR + gaze_vector[1] * sinR
        mouse_y =-gaze_vector[0] * sinR + gaze_vector[1] * cosR

        return (mouse_x, mouse_y)

    
    def display_interactive_window(self, frame):
        """
        Display using CV Window
        
        Args:
            frame: 
                   The input frame
        """

        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = ("Press '%s' key to exit" % (self.BREAK_KEY_LABELS))
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('Visualization Window', frame)

    
    def should_stop_display(self):
        """
        Check exit key from user
        """
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        
        return (key in self.BREAK_KEYS)

    @staticmethod
    def center_crop(frame, crop_size):
        """
        Center the image in the view
        """
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]
    
    @staticmethod
    def open_input_stream(path):
        """
        Open the input stream
        """
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return (cv2.VideoCapture(stream))
    
    
    @staticmethod
    def open_output_stream(path, fps, frame_size):
        """
        Open the output stream
        """
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not '.avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return (output_stream)

    def run(self, args):
        """
        Driver function trigger all the function
        Args:
        args: 
                Input args
        """
        # Open Input stream
        # We camera node is 0
        if args.input == "cam":
            path = "0"
        else:
            path = args.input

        input_stream = Mouse_Controller.open_input_stream(path)
        
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        
        # FPS init
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        
        # Get the Frame org size
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Get the frame count if its a video
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # Crop the image if the user define input W, H
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            #crop_size = (144, 240)
            frame_size = tuple(np.minimum(frame_size, crop_size))

        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        
        # Writer or CV Window
        output_stream = Mouse_Controller.open_output_stream(args.output, fps, frame_size)
        log.info("Input stream file opened")

        # Process on Input stream
        self.process(input_stream, output_stream)

        # Release Output stream if the writer selected
        if output_stream:
            output_stream.release()
        
        # Relese input stream[video or Camera node]
        if input_stream:
            input_stream.release()

        # Distroy CV Window
        cv2.destroyAllWindows()
    
    def process(self, input_stream, output_stream):
        """
        Function to capture a frame from input stream, Pre-process,
        Predict, and Display

        Args:
        input_stream: 
                The input file[Image, Video or Camera Node]
        output_stream: 
                CV writer or CV window
        """

        self.input_stream = input_stream
        self.output_stream = output_stream
        frame_count = 0
        # Loop input stream until frame is None
        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break
            
            frame_count+=1
            if self.input_crop is not None:
                frame = Mouse_Controller.center_crop(frame, self.input_crop)
            
            self.org_frame = frame.copy()

            # Get Face detection
            detections = self.frame_processor.face_detector_process(frame)
            
            # Since other three models are depend on face detection. Continue
            # only if detection happens
            if not detections:
                continue

            # Get head Position
            headPosition = self.frame_processor.head_position_estimator_process(frame)

            # Get face landmarks 
            landmarks = self.frame_processor.face_landmark_detector_process(frame)

            # Draw detection keypoints
            output = self.landmarkPostProcessing(frame, landmarks[0], detections, self.org_frame)

            gaze = self.frame_processor.gaze_estimation_process(headPosition, 
                                output[0], output[1])
            gaze_vector = gaze[0]
            
            gaze_vector = gaze_vector['gaze_vector']

            self.draw_final_result(frame, detections, headPosition, 
                                   landmarks[0], gaze_vector)
            
            if self.mc_out:
                # This count can be removed if you have high performance system
                if frame_count % 10 == 0:
                    mouse_x, mouse_y = self.get_mouse_point(headPosition, gaze_vector)
                    
                    self.mc.move(mouse_x, mouse_y)

            # Write on disk 
            if output_stream:
                output_stream.write(frame)
            
            # Display on CV Window
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break
            
            # Update FPS
            FPS = self.update_fps()
            print("[INFO] approx. FPS: {:.2f}".format(FPS))
            self.frame_num += 1