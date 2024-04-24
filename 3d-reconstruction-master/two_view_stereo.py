import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


# +
# from dataloader import load_middlebury_data
# from utils import viz_camera_poses
# -

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "We assume original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    H_i = K_i_corr @ R_irect @ np.linalg.inv(K_i)  # np.reshape(rgb_i, (3, W, H))   # [H,W,3] x [3,3] = [H,W,3]
    H_j = K_j_corr @ R_jrect @ np.linalg.inv(K_j)  # np.reshape(rgb_j, (3, W, H))   # [H,W,3] x [3,3] = [H,W,3]
    
    #transform_matrix = cv2.getPerspectiveTransform(rgb_i, rgb_i_rect)
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))   # dst
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max ,h_max))   # dst

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw):
    """
    Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_iw, R_jw : [3,3]
    T_iw, T_jw : [3,1]
        p_i = R_iw @ p_w + T_iw
        p_j = R_jw @ p_w + T_jw
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ij @ p_j + T_ij, B is the baseline
    """
    
    # right -> left :: j -> i 
    # p_right = [R_w^r inv(R_w^l)] p_left  + [-R_w^r inv(R_w^l) T_w^l  + T_w^r]
    # print('\n compute_right2left_transformation')
    
    # Correct R and T: R_ij and T_ij
    R_ij = R_iw @ R_jw.T                      # alternatively     # R_jw @ R_iw.T
    T_ij = (-R_iw) @ R_jw.T @ T_jw + T_iw     # alternatively     # (- R_jw) @ R_iw.T @ T_iw + T_jw
    
    B = np.linalg.norm(T_ij)                  # Baseline    
    
    return R_ij, T_ij, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)
    
    E = np.zeros((3,3))
    #print('\n compute_rectification_R')
    
    z_vec = np.array([0,0,1])
    row1 = np.cross(e_i, z_vec).flatten().T
    row1 = row1/np.linalg.norm(row1)
    #print('shape of row1: ', row1.shape)
    
    row2 = e_i
    #print('shape of row2: ', row2.shape)
    
    row3 = np.cross(row1, row2)
    #print('shape of row3: ', row3.shape)
    
    E[0,:] = row1
    E[1,:] = row2/np.linalg.norm(row2)
    E[2,:] = row3/np.linalg.norm(row3)
    
    R_irect = E
    

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated separately and finally summed up

    Parameters
    ---------------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    ---------------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    
    
    M = src.shape[0]
    N = dst.shape[0]                # M == N = 480
    ssd = np.zeros((M, N))          # computed for each r, g and b iteration
    
    # vectorizing by mesh_gridding the indices
    # We want for the src image N times each i going from 0...M-1
    # and for dst we want j going from 0 to N-1 repeated M times overall
    src_ind = np.arange(0,M)
    dst_ind = np.arange(0,N)
    
    iv, jv = np.meshgrid(src_ind, dst_ind, indexing='ij')   # iv and jv of shape: 480,480
    iv = iv.flatten()           # shape 230400,
    jv = jv.flatten()           # shape 230400,
    
    ssd_r = np.sum((src[iv,:,0] - dst[jv,:,0])**2, axis=1).reshape(M,N)
    ssd_g = np.sum((src[iv,:,1] - dst[jv,:,1])**2, axis=1).reshape(M,N)
    ssd_b = np.sum((src[iv,:,2] - dst[jv,:,2])**2, axis=1).reshape(M,N)
    ssd = ssd_r + ssd_g + ssd_b

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    
    M = src.shape[0]
    N = dst.shape[0]                # M == N = 480
    sad_mat = np.zeros((M, N))      # computed for each r, g and b iteration
    
    src_ind = np.arange(0,M)
    dst_ind = np.arange(0,N)
    
    iv, jv = np.meshgrid(src_ind, dst_ind, indexing='ij')   # iv and jv of shape: 480,480
    iv = iv.flatten()           # shape 230400,
    jv = jv.flatten()           # shape 230400,
    
    sad_r = np.sum(abs(src[iv,:,0] - dst[jv,:,0]), axis=1).reshape(M,N)
    sad_g = np.sum(abs(src[iv,:,1] - dst[jv,:,1]), axis=1).reshape(M,N)
    sad_b = np.sum(abs(src[iv,:,2] - dst[jv,:,2]), axis=1).reshape(M,N)
    sad = sad_r + sad_g + sad_b
    
    """
    M = src.shape[0]
    N = dst.shape[0]                # M == N = 480
    sad_mat = np.zeros((M, N))      # computed for each r, g and b iteration
    sad = np.zeros_like(sad_mat)
    for i in range(3):              # once each for the r,g and b channels
        for j in range(M):
            for k in range(N):
                sad_mat[j,k] = np.sum(abs(src[j,:,i] - dst[k,:,i]))  # sad calc
        sad += sad_mat
    """

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    
    """
    M = src.shape[0]
    N = dst.shape[0]                # M == N = 480
    zncc_mat = np.zeros((M, N))      # computed for each r, g and b iteration
    zncc = np.zeros_like(zncc_mat)
    for i in range(3):              # once each for the r,g and b channels
        src_mean = np.mean(src)
        dst_mean = np.mean(dst)
        for j in range(M):          # iterate over the left  = src image
            for k in range(N):      # iterate over the right = dst image
                # src_mean = np.mean(src[j,:,i])  # mean over the current patch of values
                # dst_mean = np.mean(dst[k,:,i])  # mean over the current patch of values
                numer = np.sum((src[j,:,i]-src_mean)*(dst[k,:,i]-dst_mean))
                denom_src = np.sqrt(np.sum((src[j,:,i]-src_mean)**2))
                denom_dst = np.sqrt(np.sum((dst[k,:,i]-dst_mean)**2))
                zncc_mat[j,k] = numer/(denom_src*denom_dst + EPS)
        zncc += zncc_mat
    """
    
    zncc = np.sum((1 / (np.expand_dims(np.std(src, axis=-2, keepdims=True), axis=1)*(np.expand_dims(np.std(dst, axis=-2, keepdims=True), axis=0)) + EPS)) * \
                  np.sum(np.expand_dims(src - np.mean(src, axis=-2, keepdims=True), axis=1) * \
                         np.expand_dims(dst - np.mean(dst, axis=-2, keepdims=True), axis=0), axis=-2, keepdims=True), axis=(-1,-2))
        

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding
    
    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """
    
    
    H = image.shape[0];    W = image.shape[1]
    
    if k_size == 1:
        patch_buffer = image.reshape(H, W, 1, 3)
    else:
        # Padding with zeros at the boundaries
        padded_img = np.zeros((H+k_size-1, W+k_size-1, 3))
        k = (k_size-1)//2
        padded_img[k:H+k,k:W+k,:] = image
        
        image_r = padded_img[:,:,0]
        image_g = padded_img[:,:,1]
        image_b = padded_img[:,:,2]
        
        patch_buf_r = np.expand_dims((np.lib.stride_tricks.sliding_window_view(image_r, (k_size,k_size), axis=(0,1)).reshape(H,W,k_size**2)), axis=-1)  # H,W,3*3,1 output
        patch_buf_g = np.expand_dims((np.lib.stride_tricks.sliding_window_view(image_g, (k_size,k_size), axis=(0,1)).reshape(H,W,k_size**2)), axis=-1)  # H,W,3*3,1 output
        patch_buf_b = np.expand_dims((np.lib.stride_tricks.sliding_window_view(image_b, (k_size,k_size), axis=(0,1)).reshape(H,W,k_size**2)), axis=-1)  # H,W,3*3,1 output
        
        patch_buffer = np.squeeze(np.stack((patch_buf_r, patch_buf_g, patch_buf_b), axis=-1))  # stack em along the last axis

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=3, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation 
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    
    
    H = rgb_i.shape[0];    W = rgb_i.shape[1]
    
    
    # converting the images into patch based representation
    patch_left  = img2patch_func(rgb_i.astype(float)/255.0, k_size)   # H,W,25,3 sized output;   k_size=5
    patch_right = img2patch_func(rgb_j.astype(float)/255.0, k_size)   # H,W,25,3 sized output
    
    # Initializing empty vectors
    disp_map = np.zeros((H,W))   # left to right: src wrt to dst
    lr_consistency_mask = np.empty((H,W))
    
    
    
    
    ##########################################################
    ################## NEW APPROACH ##########################
    
    vi_idx, vj_idx = np.arange(H), np.arange(H)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    disp_candidates.any()
    valid_disp_mask = disp_candidates > 0.0
    
    
    for u in range(W):      # iterating over across all 475 columns
        
        # Consider only the u-th columns of the left and right images
        buffer_i = patch_left[:,u,:,:]   # [H,k_size**2,3]; all rows of the u-th column from the left image
        buffer_j = patch_right[:,u,:,:]  # [H,k_size**2,3]; all rows of the u-th column from the right image
        
        # compute the kernel output for both column vectors
        value = kernel_func(buffer_i, buffer_j) # H x H output
        
        _upper = value.max() + 1.0
        
        value[~valid_disp_mask] = _upper
        
        
        # argmin of every col: best match left pixel to every right pixel
        row_ind_along_cols = np.argmin(value, axis=0).reshape(H,) 
        
        
        # argmin of every row: best match right pixel to every left pixel     
        col_ind_along_rows = np.argmin(value, axis=1).reshape(H,) 
        
        
        # Logic behind the consistency mask computation
        #for i in range(H):
        #    #step1 = i
        #    #step2 = col_ind_along_rows[step1] # This is equivalent to sequentially accessing the elements of cold_ind_along_rows
        #    #step3 = step2                     #  --- " " ---
        #    step4 = row_ind_along_cols[step3] # row_ind_along_cols[col_ind_along_rows] == np.arange(0,H) ??
        #    if step1 == step4:
        #        lr_consistency_mask[i,u] = True
        #    else:
        #        lr_consistency_mask[i,u] = False
        
        
        lr_consistency_mask[:,u] = (row_ind_along_cols[col_ind_along_rows] == np.arange(H))
        
        
        # disparity map Correct!
        disp_map[:,u] = (np.arange(0,H).reshape(H,) - col_ind_along_rows + d0)
        
    
    lr_consistency_mask = lr_consistency_mask.astype(float)
    

    return disp_map, lr_consistency_mask


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
    f = K[1,1]              # K[1,1] == K[0,0] ~= 1520
    print('f: ', f)
    H = disp_map.shape[0]
    W = disp_map.shape[1]
    
    
    # Calculating the depth map
    dep_map = (f * B) / (disp_map)       # is disp_map invertible?
    
    
    # Image center
    u0 = K[0,2]
    v0 = K[1,2]
    print('u0 and v0: ', u0, '  ', v0)
    
    
    # Initializing the point cloud matrix
    xyz_cam = np.zeros((H,W,3))
    
    X = np.expand_dims( (1/K[0,0]) * dep_map * np.repeat((np.arange(W)-u0).reshape(1,W), H, axis=0), axis=2)    # stack col vector "x" W times side by side
    print('\n\nunique values of X: ')
    print(np.unique(X))
    
    Y = np.expand_dims( (1/K[1,1]) * dep_map * np.repeat((np.arange(H)-v0).reshape(H,1), W, axis=1), axis=2)
    print('\n\nunique values of Y: ')
    print(np.unique(Y))
    
    xyz_cam = np.stack((X, Y, np.expand_dims(dep_map, axis=2)), axis=2).squeeze(3)
    print('\n\nunique values of xyz_cam: ')
    print(np.unique(xyz_cam))
    

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near = 2000,
    z_far  = 5000,):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """
    
    ## Included for normalizing
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(np.float32)

    """
    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    
    
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)
    """
    ### Using the unique values dispalyed above decide upon the values of z_near and z_far
    
    mask = mask_dep
    
    """
    mask = np.minimum(mask_dep, mask_hsv)
    
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)"""
    
    
    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)
    
    
    """
    pcl_cam = pcl_cam[mask.reshape(-1) > 0]                  # ????
    print('shape of pcl_cam: ', pcl_cam.shape)"""
    
    
    """
    o3d_pcd = o3d.geometry.PointCloud()
    
    # o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape.copy())
    
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    
    _pcl_mask[ind] = 1.0
    
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    
    mask = np.minimum(mask, mask_pcl)
    """
    
    # pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]     # Same as previous line of code....
    
    
    pcl_color = np.zeros_like(rgb.reshape(-1, 3))
    
    
    pcl_world = ((R_wc.T @ pcl_cam.T) - (R_wc.T @ T_wc)).T
    
    print('\nshape of mask     : ', mask.shape)
    print('shape of pcl_world: ', pcl_world.shape)

    
    print('\n')
    print('--------------------------------------------------------------------------------------------------')
    
    
    return mask.astype(np.float32), pcl_world.astype(np.float32), pcl_cam.astype(np.float32), pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline
    
    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj
    
    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)
    
    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "We make rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_iw,
        T_wc=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
