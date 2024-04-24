#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def point_cloud(depth_map, K):
    
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    
    u0  = K[0,2];     v0  = K[1,2]      # image center
    f_u = K[0,0];     f_v = K[1,1]      # focal lengths
    
    rows, cols = depth_map.shape[0], depth_map.shape[1]
    
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # valid_one = np.sum(depth_map > 0)
    # valid_two = np.sum(depth_map < 255)
    
    valid = (depth_map > 0) & (depth_map < 255)
    
    z = np.where(valid, depth_map / 256.0, np.nan)
    
    x = np.where(valid, z * (c - int(u0)) / f_u, 0)
    y = np.where(valid, z * (r - int(v0)) / f_v, 0)
    
    pcl = np.dstack((x, y, z))
    
    print('\nany Nans? ', np.sum(np.isnan(pcl)))
    
    print('shape of the point cloud: ', pcl.shape)
    
    return pcl


# %%
def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud
    
    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """
    
    # print('K: ', K)
    # Relationship between the depth(Z) and pixel disparity(d): Z = f B/d ##Z is inversely proportional to d ##
    f = K[1,1]                           # K[1,1] == K[0,0] ~= 3.2 mm
    H = disp_map.shape[0]
    W = disp_map.shape[1]
    
    
    # Calculating the depth map
    dep_map = (f * B) / (disp_map)       # is disp_map invertible?
    
    
    # Image center
    u0 = W // 2
    v0 = H // 2
    
    
    # Initializing the point cloud matrix
    xyz_cam = np.zeros((H,W,3))
    
    X = np.expand_dims((1/K[0,0]) * dep_map * np.repeat((np.arange(W)-u0).reshape(1,W), H, axis=0), axis=2)    # stack col vector "x" W times side by side
    
    Y = np.expand_dims((1/K[1,1]) * dep_map * np.repeat((np.arange(H)-v0).reshape(H,1), W, axis=1), axis=2)
    
    xyz_cam = np.stack((X, Y, np.expand_dims(dep_map, axis=2)), axis=2).squeeze(3)
    

    return dep_map, xyz_cam
