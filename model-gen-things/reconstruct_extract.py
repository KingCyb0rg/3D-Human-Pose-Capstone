import open3d as o3d
import numpy as np

#Functions for cleaning and measurement
from tylmen_api_functions import api_master_pipe
from model_gen_functions import cut_floor
from model_gen_functions import cloud_denoise
from model_gen_functions import generate_mesh
from data_extraction import dataExtract

def reconstruct_extract(cloud: o3d.geometry.PointCloud, noise_passes=5, waist_thresh=8, reconst_depth=7, mesh_cut=False, paint=True, silence=False):

    cut_pcd = cut_floor(cloud, silence)
    denoise_pcd = cloud_denoise(cut_pcd, noise_passes, silence)

    pcd_h, pcd_wspan, pcd_top_bot, pcd_l_r, pcd_waist, pcd_other = dataExtract(denoise_pcd, waist_thresh)
    poisson_mesh = generate_mesh(denoise_pcd, reconst_depth, mesh_cut, paint, silence)

    #print("Height: " + str(pcd_h) + "\nWingspan: " + str(pcd_wspan) + "\nBounds: " + str(pcd_top_bot) + "\nWidth Bounds:" + str(pcd_l_r))
    #o3d.visualization.draw_geometries([poisson_mesh], mesh_show_wireframe=True)


    