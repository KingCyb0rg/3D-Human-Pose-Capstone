#!/usr/bin/env python
# coding: utf-8
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
import torch
import torchvision

# Torchvision dependencies
import torch.nn.functional as nnf
from torchvision import models
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


# Reference: https://pytorch.org/vision/master/models.html


# %%
def erase_bg(imgPath, category):
    
    img = read_image(os.path.normpath(imgPath))            # ensure that the string being passed as an argument
                                                           # will be read as a path to a valid file
    
    # Step 1: Initialize model with the best available weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()                                       # set the model to evaluation mode
    
        
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
        
        
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    
    
    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    print('class_to_idx: ')
    print(class_to_idx)
    if isinstance(category, str):         # check if it is a string
        mask = normalized_masks[0, class_to_idx[category]]
    else:
        print('The catergory provided is not a string :(')
        return
    
    
    # Step 5. Interpolate to resize the masks back to the same size as the original image
    mask = torch.unsqueeze(mask, 0)
    mask = F.resize(mask, (img.shape[1], img.shape[2]), 2)
    mask = mask.detach().numpy()
    mask = mask.squeeze(0)
    mask[mask > 0.5] = int(1)
    mask[mask < 0.5] = int(0)
    
    
    return mask

