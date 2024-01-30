#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time


# %%
def sift_matching(img_i, img_j):
    
    # Initiate the OpenCV SIFT detector
    sift = cv.SIFT_create()
    
    # Find the keypoints and descriptors with SIFT
    kp_i, des_i = sift.detectAndCompute(img_i, None)
    kp_j, des_j = sift.detectAndCompute(img_j, None)
    
    
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_i, des_j, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
        
    matched_image = cv.drawMatches(img_i[:,:,::-1], kp_i, img_j[:,:,::-1], kp_j, good, None, flags=2)
    
    
    # Find the Homography matrix
    image_i = img_i.copy()
    image_j = img_j.copy()
    
    MIN_MATCH_COUNT = 10
    
    if len(good) > MIN_MATCH_COUNT:
        
        src_pts = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        H , _    = cv.findHomography( src_pts, dst_pts, cv.RANSAC, 5.0 )    # Homography matrix
        H_inv, _ = cv.findHomography( dst_pts, src_pts, cv.RANSAC, 5.0 )    # Inverse Homography matrix
        
        matchesMask = mask.ravel().tolist()
        
        h = image_i.shape[0]
        w = image_i.shape[1]
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        dst = cv.perspectiveTransform(pts, H)
        
        image_j = cv.polylines(image_j, [np.int32(dst)], True, 255, 10, cv.LINE_AA)
        
        print('\nEnough good matches found!')
        
        
    else:
        
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        
        matchesMask = None
            
        
    draw_params = dict(matchColor = (0, 255, 0),    # draw matches in red color
                       singlePointColor = None,
                       matchesMask = matchesMask,   # draw only inliers
                       matchesThickness = 10,
                       flags = 2)
    
    img3 = cv.drawMatches(image_i[:,:,::-1], kp_i, image_j[:,:,::-1], kp_j, good, None, **draw_params)
    
    plt.figure(figsize=(9,9))
    plt.imshow(img3[:,:,::-1], alpha=0.9)
    plt.title('')
    plt.show()
    
    # Reference the right half for the bounding box to be corrected later on when rectification is performed!
    # img3_right = img3[:, img3.shape[1]//2:][:,:,::-1]
    
    return H, src_pts, dst_pts 


# %%
def akaze_matching(im1, im2, index):
    
    print('\nAkaze feature matcher function invoked!')
    
    gray1 = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)             # Turn the BGR color images into gray images
    gray2 = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)             # Turn the BGR color images into gray images
    
    # initialize the AKAZE descriptor, then detect keypoints and extract local invariant descriptors 
    # from the image
    detector = cv.AKAZE_create()
    kps1, descs1 = detector.detectAndCompute(gray1, None)
    kps2, descs2 = detector.detectAndCompute(gray2, None)
    
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(descs1, descs2, 2)
    
    
    # store all the good matches as per Lowe's ratio test
    matched1 = [];    pts1 = []
    matched2 = [];    pts2 = []
    good = []
    nn_match_ratio = 0.52         # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append(cv.DMatch(len(matched1), len(matched2), 0))
            matched1.append(kps1[m.queryIdx])
            matched2.append(kps2[m.trainIdx])
            pts1.append(kps1[m.queryIdx].pt)     # returns the (u,v) values
            pts2.append(kps2[m.trainIdx].pt)     # returns the (u,v) values
    
    res = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(im1[:,:,::-1], matched1, im2[:,:,::-1], matched2, good, res)
    cv.imwrite('data/feature_matching_results/test_akaze_%d.png' %index, res)
    
    
    MIN_MATCH_COUNT = 4
    
    if len(good) >= MIN_MATCH_COUNT:
        
        src_pts = np.float32([ kps1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        H , mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0 )       # Homography matrix
        H_inv, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0 )       # Inverse Homography matrix
        
        """
        dst = H @ np.hstack((src_pts.squeeze(), np.ones((len(src_pts),1)))).T
        dst_err = np.linalg.norm(dst_pts.squeeze() - (dst/dst[2,:])[:2,:].T, axis=1)
        least_ind = np.argsort(dst_err)[:5]       # first 8 points with the smallest reproj error...
        
        H, _     = cv.findHomography( src_pts[least_ind,0,:].squeeze(), dst_pts[least_ind,0,:].squeeze(), cv.RANSAC, 5.0 )
        H_inv, _ = cv.findHomography(dst_pts[least_ind,0,:].squeeze(), src_pts[least_ind,0,:].squeeze(), cv.RANSAC, 5.0 )    # Inverse Homography matrix
        
        dst_least = np.matmul(H, np.hstack((src_pts[least_ind,0,:].squeeze(), np.ones((len(least_ind),1)))).T)
        dst_least_err = np.linalg.norm(dst_pts[least_ind,:].squeeze() - (dst_least/dst_least[2,:])[:2,:].T, axis=1)
        """
        
        """
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(len(dst_least_err)), dst_least_err, linewidth=2.15)
        plt.title('Updated reprojection with new H matrix')
        plt.show()"""
        
        plt.figure(figsize=(10,10))
        plt.title('Images {} and {}'.format(index, np.mod(index+1,8)), fontsize=17)
        plt.imshow(res), plt.show()
        
        
    else:
        
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        H, H_inv, src_pts, dst_pts, kps1, kps2, good = None, None, None, None, None, None, None
        
    
    return H, H_inv, src_pts, dst_pts, kps1, kps2, good

