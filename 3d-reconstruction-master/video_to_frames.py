#!/usr/bin/env python
# coding: utf-8
# %%

# Run this only once at the beginning, does not need to be debugged

# %%


import os
import cv2 as cv


# %%


# load a video to project something onto

def video2frames(path):

    # test path : data/meRotating.mp4
    video = cv.VideoCapture(path)
    
    success, image = video.read()
    
    dummy = print('Succesfully loaded video!') if success == True else print('Video not found :(')
    
    count = int(0)
    
    while success:
        
        cv.imwrite("data/personRotating/person_%d.png" % count, image)     # save every frame as a .png file
        
        success, image = video.read()              # read the next frame from the video loaded
        
        if not success and count == 0:
            print('Something went wrong!')
        
        elif not success and count > 0:
            print('Successfully stored the video as {} number of frames!'.format(count))
            return [not success, count]
        
        count += 1


# %%





# %%




