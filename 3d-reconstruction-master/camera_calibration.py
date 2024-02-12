#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import os
from calibrate_camera_intrinsics import initial_intrinsics, initial_extrinsics, reproj_err, filtered_new_intrinsics
from test import plot_reproj_errs, plot_scaling_ratios
from line_fitting import line_fitting as lf
import calibrate_camera_lvm as clvm


# %%
def choose_calibration_params(choice = 1):     # default choice set to hardware based pseudo calibration
    
    
    if choice == 1:
        
        """###################################################################################"""
        """    CHOICE 1. Creating the Calibration matrix "K" with the physical parameters     """
        """###################################################################################"""
        
        # Set up the variables, assuming approximate image centers == half the height and the width
        k = 401 / 25.4      # UNITS: 1/mm      # One Plus 6 has a 401 ppi screen (25.4 mm = 1 inch)
        f = 25              # UNITS: mm
        alpha_u = k * f
        alpha_v = alpha_u   # assuming the same value since all phones have the square pixels!
        
        # ASSUME the image centers for the two images [u0, v0] and [u0_prime, v0_prime] are in the center of the images
        u0 = 307.5          ;         u0_prime = u0                # Arbitrary values
        v0 = 205            ;         v0_prime = v0                # Arbitrary values
        # u0, v0             = 1000*u0/l, np.mod(1560, 1000*u0/l)  # Assumed Centers, will be corrected in the next block
        # u0_prime, v0_prime = u0       , v0                         # Assumed Centers, will be corrected in the next block
        
        
        # K and K_prime are the calibration matrices for the two images
        K       = np.array([[alpha_u,       0,   u0],
                            [      0, alpha_v,   v0],
                            [      0,       0,    1]])
        K_prime = np.array([[alpha_u,       0,   u0_prime],
                             [      0, alpha_v,   v0_prime],
                             [      0,       0,          1]])
    
    
    
    
    elif choice == 2:
        
        """###################################################################################"""
        """    CHOICE 2. Using the Calibration matrix "A" from the optimization procedure     """    
        """###################################################################################"""
        
        # 1. Load all the necessary images for custom calibration
        ref_path = os.path.normpath('data/dollar_100_ref.jpeg')      # path to the reference image
        usr_paths = [ os.path.normpath('data/bill_1.jpg'),           # the paths to the user submitted images
                      os.path.normpath('data/bill_2.jpg'),
                      os.path.normpath('data/bill_3.jpg'),
                      os.path.normpath('data/bill_4.jpg'),
                      os.path.normpath('data/bill_5.jpg') ]
        
        num_usr_imgs = len(usr_paths)
        
        
        # 2. Call the initial_intrinsics function to calculate the initial analytic intrinsic values!
        a, b, c, u0, v0, l, H_mats, world_plane, image_plane = initial_intrinsics(ref_path, usr_paths)
        
        
        # 3. With the new reprojection errors, filter the intrinsics... by re-calculating the intrinsics
        # using only those points with the smallest reprojection errors from each image/ref pair!
        a, b, c, u0, v0, l, s, s_all, ratio_err = filtered_new_intrinsics(A, R_mats, T_mats, world_plane, image_plane, num_usr_imgs)
        
        
        # 4. Construct the Intrinsic matrix with the constants obtained from the initial_intrinsics function
        A = np.array([[a, c, u0],
                      [0, b, v0],
                      [0, 0,  1]])      # The intrinsics/calibration matrix of the camera
        
        
        # 5. Calculate the scaling factors for the reprojected images - this is the missing scalar constant
        # that the reprojections need to be scaled by to align with the points on the image plane
        # These values need to be calculated only if LVM optimization is to be performed upon the intrinsics...
        updated_s_factor = []                      # list of scaling factor values
        
        for i in range(len(s)):
            reg_le = lf(s_all[i])                  # regression over the 20 points with the least reprojection error
            reg_le.horiz_line_fitting()
            updated_s_factor.append(reg_le.b)
            
        """NOTES ON "s" obtained from the filtered_new_intrinsics() function called above"""
        # s : refers to the scaling factors (on the left hand side of the equation below)
        #     s * m_tilde = A @ [R | t] @ M
        # This "s" is in fact unique to every reprojected feature point being compared to its image plane pt
        
        # Optionally apply the optimization routine to the initial intrinsics...
        # lvm = clvm.lvm_optimization(image_plane, world_plane, A, s, l, H_mats, R_mats, T_mats)   # class initialization
        # lvm.refine_intrinsics()
        
        K       = A
        K_prime = A
        
    
    
    # With the intrinsics matrix calculated using 1 of 2 techniques, we use it to calculate the corresponding
    # extrinsics for every one of the dollar bills... (not necessary)
    """
    # Calculate the Extrinsics!
    R_mats = []
    T_mats = []
        
    for i in range(num_usr_imgs):
        R, T = initial_extrinsics(K, l, H_mats[i])
        R_mats.append(R)
        T_mats.append(T)"""
    
    
    
    return K, K_prime

