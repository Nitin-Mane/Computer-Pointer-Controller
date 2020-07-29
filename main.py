#!/usr/bin/env python3

#############################################################################################################
#
#                                 Computer Pointer Controller Main Script
#
#############################################################################################################

'''
Computer Pointer Controller:
This is the main script for the running all the code and the functions. The computer pointer takes the input parameter.
Input Arguments:
        1. Model-01 -> Face Detection Model
        2. Model-02 -> Head Pose Estimation Model 
        3. Model-03 -> Landmark Detection Model 
        4. Model-04 -> Gaze Estimator Model
        5. Input (Video or Webcam) -> media file in .mp4 or CAM
        6. Device -> 'CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL'
        7. Flags -> Show if enabled option for models 
        8. Resolution -> Width and Height (Optional)

Output Arguments: 
        1. Media -> Output on screen and save file
        2. timelapse -> video time inference in fps with seconds
        3. Samples -> Output of the model outcome as per the Flags initiated
        4. perf_stats -> Statics of the inference backend
'''

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
from src.face_detection import Face_Detection
from src.head_position_estimation import Head_Pose_Estimator
from src.landmark_detection import Landmarks_Detection
from src.gaze_Estimator import Gaze_Estimation
from src.mouse_controller import Mouse_Controller_Pointer
from src.mouse_process import Mouse_Controller

# load the OpenVINO library
from openvino.inference_engine import IENetwork


# Set the Device operation types
DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

def build_argparser():
    """
    Parse command line arguments.
    -i bin/demo.mp4 
    -m_fd <path>models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml 
    -d_fd { 'CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL'}
    -o_fd 
    -m_hp <path>models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml 
    -d_hp {'CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL'}
    -o_hp 
    -m_lm <path>mo_model/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml 
    -d_lm {'CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL'}
    -o_lm 
    -m_gm <path>mo_model/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml 
    -d_gm {'CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL'}
    -o_gm 
    -o <path>results/outcome<num>
    -pc 

    :return: command line arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file in .mp4 format or enter CAM for webcam")
    
  
    parser.add_argument("-m_fd", "--model_face_detection", required=True, type=str,
                        help="Path to load an .xml file with a trained Face Detection model")               
    
    parser.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model device selection (default: %(default)s)")

    parser.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.4,
                       help="(optional) Set the Probability threshold for face detections" \
                       "(default: %(default)s)")

    parser.add_argument('-o_fd', action='store_true',
                       help="(optional) Process the face detection output")
                       
    parser.add_argument("-m_hp", "--model_head_position", required=True, type=str,
                        help="Path to load an .xml file with a trained Head Pose Estimation model")

    parser.add_argument('-d_hp', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Head Position model (default: %(default)s)")

    parser.add_argument('-o_hp', action='store_true',
                       help="(optional) Show Head Position output")

    parser.add_argument("-m_lm", "--model_landmark_regressor", required=True, type=str,
                        help="Path to load an .xml file with a trained Head Pose Estimation model") 
    parser.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")

    parser.add_argument('-o_lm', action='store_true',
                       help="(optional) Show Landmark detection output")
    
    parser.add_argument("-m_gm", "--model_gaze", required=True, type=str,
                        help="Path to an .xml file with a trained Gaze Estimation model")

    parser.add_argument('-d_gm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Gaze estimation model (default: %(default)s)")

    parser.add_argument('-o_gm', action='store_true',
                       help="(optional) Show Gaze estimation output")
    
    parser.add_argument('-o_mc', action='store_true',
                       help="(optional) Run mouse counter")

    parser.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")

    parser.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.20,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")

    parser.add_argument('-cw', '--crop_width', default=0, type=int,
                        help="(optional) Crop the input stream to this width " \
                        "(default: no crop). Both -cw and -ch parameters " \
                        "should be specified to use crop.")

    parser.add_argument('-ch', '--crop_height', default=0, type=int,
                        help="(optional) Crop the input stream to this width " \
                        "(default: no crop). Both -cw and -ch parameters " \
                        "should be specified to use crop.")
                        
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")

    parser.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")

    parser.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")

    parser.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")

    parser.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")


    return (parser)

        
##########################################################################################################

def main():
    
    args = build_argparser().parse_args()
    
    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    driverMonitoring = Mouse_Controller(args)
    driverMonitoring.run(args)


if __name__ == "__main__":
    main()