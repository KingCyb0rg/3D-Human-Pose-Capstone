#!/usr/bin/env python
# coding: utf-8
# %%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import plotly
from tqdm import tqdm
from scipy import ndimage


# Not all the frames as necessary, since the video has artifacts of the person moving about - towards and away 
# from the camera to set it up and stop the video... can AR be used to solve this issue?
# Perhaps an AP interface that instructs the person to rotate counterclockwise with an outline of a person
# that they should approximately position themselves about? If so, how do we determine when they have positioned
# themselves correctly to move on with the instructions asking them to turn??

# %%
def drop_unnecessary_frames(vpath, frameCount):
    
    dummy = [os.remove(os.path.normpath("%s/person_%d.png" % (vpath, count))) for count in range(frameCount) if ((count < 100) or (count > frameCount-100))]
    
    print('Done dropping frames!')
    
    newFrames = np.array([0 if os.path.isfile(name)==True else 1 for name in os.listdir(os.path.normpath(vpath))])
    
    numFrames = newFrames.sum()
    
    return numFrames

