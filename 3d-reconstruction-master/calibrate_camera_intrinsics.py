#!/usr/bin/env python
# coding: utf-8
# %%
import os
import os.path
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import plotly
import k3d
import glob
from tqdm import tqdm
import scipy
from scipy.linalg import solve
import time

print('Successfully imported all libraries')


# %%
# Include these in the main.ipynb notebook after converting this notebook into a .py file
# And comment it out from this notebook once converted into a .py file

ref_path = os.path.normpath('data/dollar_100_ref.jpeg')

usr_paths = [ os.path.normpath('data/bill_1.jpg'),
              os.path.normpath('data/bill_2.jpg'),
              os.path.normpath('data/bill_3.jpg'),
              os.path.normpath('data/bill_4.jpg'),
              os.path.normpath('data/bill_5.jpg') ]


# %%
def feat_matching_H(im_user, im_ref):
    
    """INPUT  : A user image and reference image pair
       OUTPUT : Homography matrix H,
                the homography mask,
                good SIFT point matches,
                user image keypoints,
                reference image keypoints"""
    
    sift = cv.SIFT_create()                                   # SIFT - detect and match features
    
    kp_user, des_user = sift.detectAndCompute(im_user, None)  # find the keypoints and descriptors with SIFT
    kp_ref , des_ref = sift.detectAndCompute(im_ref, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_user, des_ref, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)                                     # Store all the good matches as per Lowe's ratio test.
    
    matched_image = cv.drawMatches(im_user, kp_user, im_ref, kp_ref, good, None, flags=2)
    
    image_user = im_user.copy()
    image_ref  = im_ref.copy()
    
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        
        usr_pts = np.float32([ kp_user[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        ref_pts = np.float32([ kp_ref[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        H , mask = cv.findHomography(usr_pts, ref_pts, cv.RANSAC, 5.0)    # Homography matrix
        H_inv, _ = cv.findHomography(ref_pts, usr_pts, cv.RANSAC, 5.0)    # Inverse Homography matrix
        
    return H, mask, good, kp_user, kp_ref, usr_pts, ref_pts


# %%
def draw_matches(user_100, ref_100, H, mask, good, kp_user, kp_ref):
    
    
    """INPUT: A user image, reference image pair,
              the Homography matrix that takes us from the user submitted picture to the reference image,
              the Homography mask,
              good matches that the SIFT detector computed in feat_matching_H,
              user image keypoints - from the SIFT detector
              reference image keypoints - from the SIFT detector
       OUTPUT: A plot that shows green polylines connecting matching points between the user submitted
               image and the reference image"""
    
    
    matchesMask = mask.ravel().tolist()
    
    h = user_100.shape[0]                              # user_100 one of the submitted pictures
    w = user_100.shape[1]                              # user_100 one of the submitted pictures
    
    pts = np.float32([[0, 0],[0, h-1],[w-1, h-1],[w-1, 0]]).reshape(-1,1,2)
    
    dst = cv.perspectiveTransform(pts, H)
    
    ref_100 = cv.polylines(ref_100, [np.int32(dst)], True, 255, 10, cv.LINE_AA)
    
    draw_params = dict(matchColor = (0, 255, 0),       # draw matches in red color
                       singlePointColor = None,
                       matchesMask = matchesMask,      # draw only inliers
                       matchesThickness = 10,
                       flags = 2)

    img3 = cv.drawMatches(user_100, kp_user, ref_100, kp_ref, good, None, **draw_params)
    
    plt.figure(figsize=(12,12))
    plt.imshow(img3[:,:,::-1], alpha=0.9),plt.show()
    plt.show()
    
    user_pts = np.float32([ kp_user[m.queryIdx].pt for m in good ]).reshape(-1,2)
    ref_pts  = np.float32([ kp_ref[m.trainIdx].pt for m in good ]).reshape(-1,2)


# %%
def v_ij(H, i, j):
    
    
    """INPUT: The Homography matrix between a user image and the reference image,
              i and j column indices of the H matrix being passed
       OUTPUT: a 2 x 6 matrix that can be appended to the other rows of a final V used to solve for b"""
    
    
    vij =[ H[i][0]*H[j][0], 
           H[i][0]*H[j][1] + H[i][1]*H[j][0], 
           H[i][1]*H[j][1],
           H[i][2]*H[j][0] + H[i][0]*H[j][2],
           H[i][2]*H[j][1] + H[i][1]*H[j][2],
           H[i][2]*H[j][2] ]
    
    return np.asarray(vij)


# %%
def initial_intrinsics(ref_path, usr_paths, drawMatches=False):
    
    """
    # 1. Load the image of a 100 dollar bill being used as Reference"""
    ref_img = cv.imread(ref_path)
    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2RGB)
    
    """
    # 2. Scaling the reference Dollar Bill by the right amount to match the physical dimensions 
         of a real bill"""
    ref_100 = cv.resize(ref_img, (1560, 665), interpolation = cv.INTER_LINEAR)
    
    """
    # 2. Load the user submitted images"""
    numImg   = len(usr_paths)
    user_100 = []                            # initialize an empty list for all the user submitted images
    H_mats   = []
    masks    = []
    good_matches = []
    kp_refs  = []
    kp_users = []
    world_plane = []
    image_plane = []
    
    for i, path in enumerate(usr_paths):
        
        print('\nProcessing user submitted image ', i+1)
        
        # Load the path of every user submitted image and 
        user_100_ith = cv.imread(os.path.normpath(usr_paths[i]))
        user_100.append(cv.cvtColor(user_100_ith, cv.COLOR_BGR2RGB))
        
        H, mask, good, kp_user, kp_ref, usr_pts, ref_pts = feat_matching_H(user_100_ith, ref_100)
        
        usr_pts = np.squeeze(usr_pts, axis=1)   # to convert Nx1x2 into Nx2 arrays
        ref_pts = np.squeeze(ref_pts, axis=1)   # to convert Nx1x2 into Nx2 arrays
        
        H_mats.append(H)
        masks.append(mask)
        good_matches.append(good)
        kp_refs.append(kp_ref)
        kp_users.append(kp_user)
        
        world_plane.append(usr_pts)
        image_plane.append(ref_pts)
    
    
    """
    # 3. Draw matches for the first and the last user submitted pictures..."""
    if drawMatches == True:
        for i in range(numImg):
            print('User Submitted Image: ', int(i))
            draw_matches(user_100[i], ref_100, H_mats[i], masks[i], 
                         good_matches[i], kp_users[i], kp_refs[i])
            print('\n\n')
    
    
    """
    # 4. Create the "V" matrix
    # 4. a) Initialize the "V" matrix with the 0-th user submitted image's corresponding V matrix rows
    #       Reference: Flexible Camera Calibration By Viewing a Plane From Unknown Orientations,
    #       Zhengyou Zhang, 1999."""
    V = np.vstack((v_ij(H_mats[0], 1, 2), v_ij(H_mats[0], 1, 1)-v_ij(H_mats[0], 2, 2)))
    
    # 4. b) Append the other images' V matrix rows to the initialzed 2 x 6 matrix
    for i in range(1, numImg):
        V = np.vstack((V, np.vstack((v_ij(H_mats[i], 1, 2), v_ij(H_mats[i], 1, 1)-v_ij(H_mats[i], 2, 2)))))
    
    
    """
    # 5. Setting up the linear matrix equation to be solved for the initial set of "b" vector elements"""
    c = np.zeros((V.shape[0],1))                         # Right Hand side
    b = np.asarray(np.linalg.lstsq(V, c, rcond=None))
    
    
    """
    # 6. Designate the elements of the solved "b" vector as their rightful equivalent elements
    # of the B matrix where B = A@A where A is the intrinsics/calibration matrix"""
    b = np.asarray(b)
    B11 = b[3][0]
    B12 = b[3][1]
    B22 = b[3][2]
    B13 = b[3][3]
    B23 = b[3][4]
    B33 = b[3][5]
    
    
    """
    # 7. Compute the initial estimates of the intrinsics"""
    v0    = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(lamda / B11)
    beta  = np.sqrt(lamda * B11 / (B11*B22 - B12**2))
    c     = -B12 * alpha**2 * beta / lamda
    u0    = c*v0 / alpha - B13*alpha**2 / lamda
    
    
    print('\nIntrinsics: \n')
    print('Image center (u0, v0)  : (', u0, ',', v0, ')')
    print('Scaling Factor lambda: ', lamda)
    print('alpha : ', alpha)
    print('beta  : ',  beta)
    print('c     : ',     c)
    
    return alpha, beta, c, u0, v0, lamda, H_mats, world_plane, image_plane


# %%
def initial_extrinsics(A, lam, H):
    
    """ Reference: "Flexible Camera Calibration By Viewing a Plane From Unknown Orientations",
                    by Zhang 1999 - Pg 13
        This function calculates the extrinsics given the intrinsics matrix A, the lambda factor (also
        from the intrinsics function), and the Homography matrix of the image being dealt with"""
    
    A_inv = np.linalg.inv(A)
    r1 = lam * A_inv @ H[:,0]
    r2 = lam * A_inv @ H[:,1]
    r3 = np.cross(r1, r2)
    t  = lam * A_inv @ H[:,2]
    
    R = np.vstack((r1, r2, r3)).T
    
    return R, t


# %%
def reproj_err(A, R_mats, T_mats, wPts, iPts, numIms):
    
    err_vecs = []
    wi_proj = []
    ratio_of_errs = []
    
    for i in range(numIms):    # for one image at a time
        
        wi = wPts[i]
        wi = np.hstack((wi, np.ones((len(wi), 1))))   # Turn Nx2 --> Nx3
        
        wi_p = (A @ (R_mats[i] @ wi.T + np.expand_dims(T_mats[i], axis=1))).T   # Project onto img plane
        
        wi_p = wi_p[:,:2] / np.expand_dims(wi_p[:,2], axis=1)       # scale element wise by 3rd col values - to de-homogenize
        
        scaling_ratio = np.expand_dims(np.linalg.norm(wi_p, axis=1)/np.linalg.norm(np.asarray(iPts[i]), axis=1), axis=1)
        ratio_of_errs.append(scaling_ratio)
        
        wi_p = wi_p / scaling_ratio
        
        e = np.linalg.norm(np.asarray(iPts[i]) - wi_p, axis=1)      # Error calculated after the scale is adjusted for
        
        err_vecs.append(e)                                          # error vectors appended to a bigger list
        
        wi_proj.append(wi_p)                                        # correctly scaled re-projected world plane points
        
        
    return err_vecs, wi_proj, ratio_of_errs


# %%
def filtered_new_intrinsics(A, R_mats, T_mats, wPts, iPts, numIms):
    
    """Calling the reprojection error function and using the errors to filter out the better
       matches (low error points) in order to refine and re-calculate the initial intrinsics"""
    
    err, wi_proj, ratio_err = reproj_err(A, R_mats, T_mats, wPts, iPts, numIms)
    
    
    w_filtered = [];    i_filtered = [];    H_new      = []
    V          = np.zeros((2*numIms, 6))
    s          = []                         # scaling factors (averaged) for each image
    s_all      = []
    
    for i in range(numIms):
        
        # filtering out the first 15 sets of points with smallest reprojection errors!!
        best_inds = np.argsort(err[i])[:20]     # argsort: returns the indices of those points with the smallest errors!!
        
        w_filtered_i = wPts[i][best_inds, :]
        i_filtered_i = iPts[i][best_inds, :]
        
        H, _ = cv.findHomography(w_filtered_i, i_filtered_i, cv.RANSAC, 5.0)    # New Homography matrix
    
        H_new.append(H)
        
        V[i*2:i*2+2,:] = np.vstack((v_ij(H, 1, 2), v_ij(H, 1, 1)-v_ij(H, 2, 2)))
        
        s_i = ratio_err[i][best_inds]
        s.append(np.mean(s_i))
        s_all.append(s_i)
    
    
    # Setting up the Vb = c linear matrix equation
    c = np.zeros((V.shape[0],1))                               # Right Hand side vector "c"
    b = np.asarray(np.linalg.lstsq(V, c, rcond=None))      # vector of unknowns "b"
        
    B11 = b[3][0];        B12 = b[3][1];        B22 = b[3][2]
    B13 = b[3][3];        B23 = b[3][4];        B33 = b[3][5]
    
    
    # New refined intrinsics!
    v0    = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(lamda / B11)
    beta  = np.sqrt(lamda * B11 / (B11*B22 - B12**2))
    c     = -B12 * alpha**2 * beta / lamda
    u0    = c*v0 / alpha - B13*alpha**2 / lamda
        
        
    print('\nNew, slightly refined Intrinsics based on filtered points after calculating\
    reprojection errors: \n')
    print('\nImage center (u0, v0)  : (', u0, ',', v0, ')')
    print('Scaling Factor lambda: ', lamda)
    print('alpha : ', alpha)
    print('beta  : ',  beta)
    print('c     : ',     c)
    
    
    return alpha, beta, c, u0, v0, lamda, s, s_all, ratio_err

# %%
# How to call the initial_intrinsics function in main... 
# with available paths for the user submitted images and the reference dollar bill image

# a, b, c, u, v, l = initial_intrinsics(ref_path, usr_paths)

