import open3d as o3d
import numpy as np

#Functions for cleaning and measurement
from ModelGenFunctions import cut_floor
from ModelGenFunctions import cloud_denoise
from ModelGenFunctions import generate_mesh
from data_extraction import dataExtract

#
# I/O will go here when i can see what i'm doing on the ec2 instance.
#

def reconstruct_extract(cloud: o3d.geometry.PointCloud, noise_passes=5, waist_thresh=8, reconst_depth=7, mesh_cut=False, paint=True, silence=False):

    #TEMP
    #FIX THE PATH FOR YOUR DIRECTORY
    path = './GitHub/3D-Human-Pose-Capstone/model-gen-things/Point Clouds/set 1/brandon slow.ply'
    pcd = o3d.io.read_point_cloud(path)
    #TEMP END


    #basic error handling, if issues start happening add more for diagnosis
    if(pcd.has_points() == False):
        print("There was a problem with the point cloud: Cloud is Empty.")
        exit(1)

    cut_pcd = cut_floor(pcd, silence)
    denoise_pcd = cloud_denoise(cut_pcd, noise_passes, silence)
    pcd_height, pcd_wspan, pcd_heightbounds, pcd_widthbounds = dataExtract(denoise_pcd, waist_thresh)
    poisson_mesh = generate_mesh(denoise_pcd, reconst_depth, mesh_cut, paint, silence)

    print("Height: " + str(pcd_height) + "\nWingspan: " + str(pcd_wspan) + "\nBounds: " + str(pcd_heightbounds) + "\nWidth Bounds:" + str(pcd_widthbounds))
    o3d.visualization.draw_geometries([poisson_mesh], mesh_show_wireframe=True)
    