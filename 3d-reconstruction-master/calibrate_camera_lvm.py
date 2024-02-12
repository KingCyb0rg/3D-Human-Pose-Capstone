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

from line_fitting import line_fitting as lf


# %%
class lvm_optimization:
    
    
    def __init__(self, img_plane, world_plane, A, s, lamda, H_mats, R_mats, T_vecs):
        
        self.numImgs = len(H_mats)           # Number of user submitted images = length of the H_mats list
        self.H = H_mats                      # The Homography matrices
        self.R = R_mats                      # The rotation matrices from H_mats
        self.T = T_vecs                      # The translation vectors from H_mats
        
        # image plane points/world plane points on the reference dollar bill/user submitted dollar bill
        # East of the self.numImgs number of list elements in img_plane and world_plane is a Nx2 array
        self.img_plane = img_plane           # self.numImgs number of elements in a list
        self.world_plane = world_plane       # self.numImgs number of elements in a list
        
        self.s = s                           # Scale factors (one for each of the images...)
        self.A = A                           # / np.mean(self.s)         # Scaling A by s
        self.lamda = lamda
        
        self.step = 0.0001                   # initial LVM step size
        
        print('\nClass lvm_optimization initialized!')
    
    
    
    def calculate_extrinsics(self):
    
        for i in range(self.numImgs):
            
            A_inv = np.linalg.inv(self.A)
            r1    = self.lamda * A_inv @ self.H[i][:,0]        # lamda (obtained from the initial intrinsics output)
            r2    = self.lamda * A_inv @ self.H[i][:,1]        
            r3    = np.cross(r1, r2)                           
            self.T[i] = self.lamda * A_inv @ self.H[i][:,2]    # over-writing the import T vectors after having been updated with the updated self.A matrix
            self.R[i] = np.vstack((r1, r2, r3)).T              # over-writing the import R matrices after having been updated with the updated self.A matrix
    
    
    
    def project_world_pts(self):
        
        m_proj = []
        
        self.calculate_extrinsics()
        
        for i in range(self.numImgs):
            
            w       = np.asarray(self.world_plane[i])
            
            wrld_pl = np.hstack((np.asarray(self.world_plane[i]), np.ones((len(self.world_plane[i]),1))))
            
            proj    = self.A @ (self.R[i] @ wrld_pl.T + np.expand_dims(self.T[i], axis=1))      # [3x3] @ [3x3] @ [3xN]
            
            proj    = (proj / proj[2,:])[:2,:]      # Un-homogenizing the points --> normalizing wrt z coordinates and removing the 3rd column!
            
            # projection of the world points onto the image plane
            m_proj.append(proj.T)
        
        
        self.mHat = m_proj
        
        
        """
        # plot the projected points and how different they are from the image plane points
        f, ax = plt.subplots(3, 2, figsize=(10,15))
        
        ax[0,0] = plt.subplot(3,2,1)
        ax[0,0].scatter(self.img_plane[0][:,0], self.img_plane[0][:,1], s=np.ones_like(self.img_plane[0][:,0])*10, c='k', marker='o') 
        ax[0,0].scatter(self.mHat[0][:,0], self.mHat[0][:,1], s=np.ones_like(self.mHat[0][:,0])*20, c='r', marker='x')
        
        ax[0,1] = plt.subplot(3,2,2)
        ax[0,1].scatter(self.img_plane[1][:,0], self.img_plane[1][:,1], s=np.ones_like(self.img_plane[1][:,0])*10, c='k', marker='o') 
        ax[0,1].scatter(self.mHat[1][:,0], self.mHat[1][:,1], s=np.ones_like(self.mHat[1][:,0])*20, c='r', marker='x')
        
        ax[1,0] = plt.subplot(3,2,3)
        ax[1,0].scatter(self.img_plane[2][:,0], self.img_plane[2][:,1], s=np.ones_like(self.img_plane[2][:,0])*10, c='k', marker='o') 
        ax[1,0].scatter(self.mHat[2][:,0], self.mHat[2][:,1], s=np.ones_like(self.mHat[2][:,0])*20, c='r', marker='x')
        
        ax[1,1] = plt.subplot(3,2,4)
        ax[1,1].scatter(self.img_plane[3][:,0], self.img_plane[3][:,1], s=np.ones_like(self.img_plane[3][:,0])*10, c='k', marker='o') 
        ax[1,1].scatter(self.mHat[3][:,0], self.mHat[3][:,1], s=np.ones_like(self.mHat[3][:,0])*20, c='r', marker='x')
        
        ax[2,0] = plt.subplot(3,2,5)
        ax[2,0].scatter(self.img_plane[4][:,0], self.img_plane[4][:,1], s=np.ones_like(self.img_plane[4][:,0])*10, c='k', marker='o') 
        ax[2,0].scatter(self.mHat[4][:,0], self.mHat[4][:,1], s=np.ones_like(self.mHat[4][:,0])*20, c='r', marker='x')
        """
        
    
    
    def cost(self):
        
        # Squared difference between the image plane points and the projected ones
        m = self.img_plane
        m_proj = self.mHat
        Xsq = 0                                             # initial cost function val
        
        self.project_world_pts()
        
        for i in range(self.numImgs):
            m_sq = np.einsum('...i, ...i', m[i], m[i])                # m has to be Nx3
            mHat_sq = np.einsum('...i, ...i', m_proj[i], m_proj[i])   # m_proj has to be Nx3
        
        Xsq = abs(np.sum(m_sq - mHat_sq))                        # scalar sum of the element wise difference
                                                            # between two Nx1 vectors
        return Xsq
            
    
    
    
    def compute_jacobian_beta(self):       # REF: Equation 15.5.6 "beta_k", Pg 800, Numerical Recipes
        
        beta = np.zeros(6)                         # grad vector (acc to Numerical Recipes)
        m      = self.img_plane
        m_proj = self.world_plane
        
        for i in range(0, self.numImgs):
            
            m_proj_i = np.hstack((np.asarray(m_proj[i]), np.ones((len(np.asarray(m_proj[i])),1))))         # N x 3 array
            
            xyi_H0 = m_proj_i @ np.expand_dims(self.H[i][0,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            xyi_H1 = m_proj_i @ np.expand_dims(self.H[i][1,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            xyi_H2 = m_proj_i @ np.expand_dims(self.H[i][2,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            
            
            beta[0] += -2 * (self.A[0,0] * np.sum(xyi_H0**2)         + \
                             self.A[0,1] * np.sum((xyi_H0 * xyi_H1)) + \
                             self.A[0,2] * np.sum((xyi_H0 * xyi_H2))
                            ) / (self.A[2,2]**2)
            
            beta[1] += -2 * (self.A[1,1]*np.sum(xyi_H1**2)           + \
                             self.A[1,2]*np.sum(xyi_H1 * xyi_H2)
                            ) / (self.A[2,2]**2)
            
            beta[2] += -2 * (self.A[0,1] * np.sum(xyi_H1**2)         + \
                             self.A[0,0] * np.sum((xyi_H0 * xyi_H1)) + \
                             self.A[0,2] * np.sum((xyi_H1 * xyi_H2))
                            ) / (self.A[2,2]**2)
            
            beta[3] += -2 * (self.A[0,2] * np.sum(xyi_H2**2)         + \
                             self.A[0,1] * np.sum((xyi_H1 * xyi_H2)) + \
                             self.A[0,0] * np.sum((xyi_H0 * xyi_H2))
                            ) / (self.A[2,2]**2)
            
            beta[4] += -2 * (self.A[1,2] * np.sum(xyi_H1**2)         + \
                             self.A[1,1] * np.sum(xyi_H1 * xyi_H2)
                            ) / (self.A[2,2]**2)
            
            beta[5] += -2 * (self.A[0,0]**2 * np.sum(xyi_H0**2)     + \
                             self.A[0,1]**2 * np.sum(xyi_H1**2)     + \
                             self.A[0,2]**2 * np.sum(xyi_H2**2)     + \
                             2 * self.A[0,0]*self.A[0,1] * np.sum(xyi_H0 * xyi_H1)  + \
                             2 * self.A[0,1]*self.A[0,2] * np.sum(xyi_H1 * xyi_H2)  + \
                             2 * self.A[0,0]*self.A[0,2] * np.sum(xyi_H0 * xyi_H2)  + \
                             self.A[1,1]**2 * np.sum(xyi_H1**2)     + \
                             self.A[1,2]**2 * np.sum(xyi_H2**2)     + \
                             2 * self.A[1,1]*self.A[1,2] * np.sum(xyi_H1 * xyi_H2)  + \
                             np.sum(xyi_H2**2)
                            ) / (self.A[2,2]**3)
        
        self.jacobian = beta
    
    
    
    
    def compute_hessian_alpha(self):
        
        alpha = np.zeros((6,6))
        m      = self.img_plane
        m_proj = self.world_plane
        
        for i in range(0, self.numImgs):
            
            m_proj_i = np.hstack((np.asarray(m_proj[i]), np.ones((len(np.asarray(m_proj[i])),1))))         # N x 3 array
            
            xyi_H0 = m_proj_i @ np.expand_dims(self.H[i][0,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            xyi_H1 = m_proj_i @ np.expand_dims(self.H[i][1,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            xyi_H2 = m_proj_i @ np.expand_dims(self.H[i][2,:], axis=0).T   # Nx3 @ 3x1 = Nx1
            
            
            alpha[0,0] = -2 * np.sum(xyi_H0**2) / self.A[2,2]
            alpha[0,1] =  0
            alpha[0,2] = -2 * np.sum(xyi_H0 * xyi_H1) / self.A[2,2]
            alpha[0,3] = -2 * np.sum(xyi_H0 * xyi_H2) / self.A[2,2]
            alpha[0,4] =  0
            alpha[0,5] = -4 * (self.A[0,0] * np.sum(xyi_H0**2)          + \
                               self.A[0,1] * np.sum(xyi_H0 * xyi_H1)    + \
                               self.A[0,2] * np.sum(xyi_H0 * xyi_H2)
                               ) / (self.A[2,2]**3)
            
            alpha[1,0] =  0
            alpha[1,1] = -2 * np.sum(xyi_H1**2) / self.A[2,2]
            alpha[1,2] =  0
            alpha[1,3] =  0
            alpha[1,4] = -2 * np.sum(xyi_H1 * xyi_H2) / self.A[2,2]
            alpha[1,5] = -4 * (self.A[1,1]*np.sum(xyi_H1**2)            + \
                               self.A[1,2]*np.sum(xyi_H1 * xyi_H2)
                               ) / (self.A[2,2]**2)
            
            alpha[2,0] = -2 * np.sum(xyi_H0 * xyi_H1) / (self.A[2,2]**2)
            alpha[2,1] =  0
            alpha[2,2] = -2 * np.sum(xyi_H1**2) / (self.A[2,2]**2)
            alpha[2,3] = -2 * np.sum(xyi_H1 * xyi_H2) / (self.A[2,2]**2)
            alpha[2,4] =  0
            alpha[2,5] =  -4 * (self.A[0,1] * np.sum(xyi_H1**2)         + \
                                self.A[0,0] * np.sum(xyi_H0 * xyi_H1)   + \
                                self.A[0,2] * np.sum(xyi_H1 * xyi_H2)
                                ) / (self.A[2,2]**3)
            
            alpha[3,0] = -2 * np.sum(xyi_H0 * xyi_H2) / (self.A[2,2]**2)
            alpha[3,1] =  0
            alpha[3,2] = -2 * np.sum(xyi_H1 * xyi_H2) / (self.A[2,2]**2)
            alpha[3,3] = -2 * np.sum(xyi_H2**2) / (self.A[2,2]**2)
            alpha[3,4] =  0
            alpha[3,5] = -4 * (self.A[0,2] * np.sum(xyi_H2**2)         + \
                               self.A[0,1] * np.sum(xyi_H1 * xyi_H2)   + \
                               self.A[0,0] * np.sum(xyi_H0 * xyi_H2)
                               ) / (self.A[2,2]**3)
                                    
            alpha[4,0] =  0
            alpha[4,1] = -2 * np.sum(xyi_H1 * xyi_H2) / (self.A[2,2]**2)
            alpha[4,2] =  0
            alpha[4,3] =  0
            alpha[4,4] = -2 * np.sum(xyi_H2**2) / (self.A[2,2]**2)
            alpha[4,5] = -4 * (self.A[1,2] * np.sum(xyi_H1**2)         + \
                               self.A[1,1] * np.sum(xyi_H1 * xyi_H2)     \
                               ) / (self.A[2,2]**2)
            
            alpha[5,0] = -2 * (2 * self.A[0,0] * np.sum(xyi_H0**2)         + \
                               2 * self.A[0,1] * np.sum(xyi_H0 * xyi_H1)   + \
                               2 * self.A[0,2] * np.sum(xyi_H0 * xyi_H2)     \
                               ) / (self.A[2,2]**3)
            alpha[5,1] = -2 * (2 * self.A[1,1] * np.sum(xyi_H1**2)         + \
                               2 * self.A[1,2] * np.sum(xyi_H1 * xyi_H2)     \
                               ) / (self.A[2,2]**3)
            alpha[5,2] =  -2 * (2 * self.A[0,1] * np.sum(xyi_H1**2)        + \
                                2 * self.A[0,0] * np.sum(xyi_H0 * xyi_H1)  + \
                                2 * self.A[0,2] * np.sum(xyi_H1 * xyi_H2)    \
                                ) / (self.A[2,2]**3)
            alpha[5,3] =  -2 * (2 * self.A[0,2] * np.sum(xyi_H2**2)        + \
                                2 * self.A[0,1] * np.sum(xyi_H1 * xyi_H2)  + \
                                2 * self.A[0,0] * np.sum(xyi_H0 * xyi_H2)    \
                                ) / (self.A[2,2]**3)
            alpha[5,4] =  -2 * (2 * self.A[1,2] * np.sum(xyi_H2**2)        + \
                                2 * self.A[1,1] * np.sum(xyi_H1 * xyi_H2)    \
                                ) / (self.A[2,2]**3)
            alpha[5,5] =  -6 * (self.A[0,0]**2 * np.sum(xyi_H0**2)                         + \
                                (self.A[0,1]**2 + self.A[1,1]**2) * np.sum(xyi_H1**2)      + \
                                (self.A[0,2]**2 + self.A[1,2]**2 + 1) * np.sum(xyi_H2**2)  + \
                                2 * self.A[0,0]*self.A[0,1] * np.sum(xyi_H0 * xyi_H1)      + \
                                2 * self.A[0,0]*self.A[0,2] * np.sum(xyi_H0 * xyi_H2)      + \
                                2 * (self.A[1,1]*self.A[1,2] + self.A[0,1]*self.A[0,2])    * \
                                np.sum(xyi_H1 * xyi_H2)) / (self.A[2,2]**4)
        
        self.hessian = alpha
        
                               
            
    
    def solve_for_delta(self):
        
        # retrieve the gradient beta vectos
        # retrieve the alpha hessian matrix
        
        self.compute_hessian_alpha()
        self.compute_jacobian_beta()
        
        nC = np.max([np.max(self.hessian), np.max(self.jacobian)])
        
        if (nC) == 0.0:
            nC = np.min([np.min(self.hessian), np.min(self.jacobian)])
            self.hessian  = self.hessian  / nC
            self.jacobian = self.jacobian / nC
        else: 
            self.hessian  =  self.hessian  / nC
            self.jacobian =  self.jacobian / nC
            
        
        alpha_prime = self.hessian + self.step * np.diag(np.diag(self.hessian))
        
        print('alpha_prime: ')
        print(alpha_prime)
        
        print('\nJacobian: ')
        print(self.jacobian)
        
        if np.any(np.isnan(alpha_prime)) or np.any(np.isinf(alpha_prime)):
            self.flag = 1
            return
        elif np.any(np.isnan(self.jacobian)) or np.any(np.isinf(self.jacobian)):
            self.flag = 1
            return
        
        self.deltaA = solve(alpha_prime, self.jacobian)         # obtain the delta vector for update
    
    
    
    
    def updateA(self):
        
        self.A[0,0] += self.deltaA[0]
        self.A[1,1] += self.deltaA[1]
        self.A[0,1] += self.deltaA[2]
        self.A[0,2] += self.deltaA[3]
        self.A[1,2] += self.deltaA[4]
        self.A[2,2] += self.deltaA[5]
        
        
        
    def undo_updateA(self):
        
        self.A[0,0] -= self.deltaA[0]
        self.A[1,1] -= self.deltaA[1]
        self.A[0,1] -= self.deltaA[2]
        self.A[0,2] -= self.deltaA[3]
        self.A[1,2] -= self.deltaA[4]
        self.A[2,2] -= self.deltaA[5]
    
        
    
    def refine_intrinsics(self):
        
        delta_cost = 1
        
        self.project_world_pts()               # Step 1
        
        Xsq_A = self.cost()                    # Step 2: compute cost with current A intrinsics
        
        cost_over_time = [Xsq_A]               # Initial cost function
        
        it = 0                                 # iteration count
        
        self.flag = 0
        
        # Begin iterations!
        while delta_cost > 10**(-12):          # (delta_cost > 10**(-6)) - original condition
            
            self.solve_for_delta()             # Calculate self.deltaA
            
            if self.flag:
                break
            
            self.updateA()                     # Update matrix A with the new delta calculated
            
            self.project_world_pts()           # Project the world points with the updated intrinsics
            
            Xsq_A_new = self.cost()            # Use the new A to compute the new cost
            
            
            if Xsq_A_new >= Xsq_A:             # Step size change and update-rules
                self.step *= 2                 # 
                self.undo_updateA()            # 
            
            else:
                self.step /= 2                 # 
            
            delta_cost = Xsq_A_new - Xsq_A     # computing the change in the cost function
            
            # cost_over_time.append(Xsq_A_new) # creating a list of the cost functions over iterations 
            
            Xsq_A = Xsq_A_new                  # assign the new cost to Xsq_A old cost value
            
            it += 1
            
            time.sleep(0.1)
        
        
        print('\n\n\n')
        print('-------------------------------------------------------------------------------------')
        print('\n\n\nRefined Intrinsics: ')
        print('\n\nRefined Intrinsics: ')
        print('a  : {:.8f} mm'.format(1000 * self.A[0,0]/self.A[2,2]))              # focal length in millimeters along u axis
        print('b  : {:.8f} mm'.format(1000 * self.A[1,1]/self.A[2,2]))              # focal length in millimeters along v axis
        print('c  : {:.8f}  x'.format(1000 * self.A[0,1]/self.A[2,2]))              # skewness factor
        print('u0 : {:.8f} pixels'.format(1000*self.A[0,2]/self.A[2,2]))                 # in pixels based on reference image scale: 1pix==1mm
        print('v0 : {:.8f} pixels'.format(np.mod(1000*self.A[1,2]/self.A[2,2], 1560)))   # in pixels based on reference image scale: 1pix==1mm
        
        self.a = self.A[0,0]/self.A[2,2]
        self.b = self.A[1,1]/self.A[2,2]
        self.c = self.A[0,1]/self.A[2,2]
        
        self.u0 = 1000*self.A[0,2]/self.A[2,2]
        self.v0 = np.mod(1000*self.A[1,2]/self.A[2,2], 1560)

