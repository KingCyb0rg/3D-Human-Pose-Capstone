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

#Variable/Parameter Assignment Block
silence = False # silences console output from supported functions.
noise_passes = 5 # denoise passes: decrease to 4 if there are holes, increasing past 7 likely to completely decimate model
waist_thresh = 8 # ???
reconst_depth = 7 # reconstruction depth: amount of subdivisions. definitely exponential complexity, advise going no higher than 10 or 11
mesh_cut = False # experimental floor cutting for the mesh to remove "drape", no time for detailed refinement
paint = True # paint mesh a solid color instead of keeping color information from the point cloud.

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