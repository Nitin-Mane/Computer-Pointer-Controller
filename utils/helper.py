#!/usr/bin/env python3

#########################################################################################################
#
#                                        Helper Script
#
########################################################################################################
'''
Helper Script: 

This script is processing technique for the ROI region to map in the frame. The frame are process numerically and based on the taget shape the dimension are set. 
The default batch is set as 1 and the frame are pre-processed for better outcome.
'''
# load libary
import cv2 # Load OpenCV library
import numpy as np # load numerical operation library
from numpy import clip # load clip library

def cut_roi(frame, roi):
    # set the position of the ROI region
    p1 = roi.position.astype(int)
    p1 = clip(p1, [0, 0], [frame.shape[-1], frame.shape[-2]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = clip(p2, [0, 0], [frame.shape[-1], frame.shape[-2]])
    # frame array pointers
    return (np.array(frame[:, :, p1[1]:p2[1], p1[0]:p2[0]]))

def cut_rois(frame, rois):
    # cut the ROI in the frame
    return [cut_roi(frame, roi) for roi in rois]

def resize_input(frame, target_shape):
    # the frame are outline with the target shape. 
    # Target shape dimensions
    assert len(frame.shape) == len(target_shape), \
        "Expected a frame with %s dimensions, but got %s" % \
        (len(target_shape), len(frame.shape))

    assert frame.shape[0] == 1, "Only batch size 1 is supported"
    n, c, h, w = target_shape

    input = frame[0]
    # process the target shape income and the transpose it.
    if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
        input = input.transpose((1, 2, 0)) # process to HWC
        input = cv2.resize(input, (w, h))
        input = input.transpose((2, 0, 1)) # process to CHW

    return (input.reshape((n, c, h, w)))