import open3d as o3d
import numpy as np

from ModelGenFunctions import cut_floor
from ModelGenFunctions import cloud_denoise
from ModelGenFunctions import generate_mesh
from data_extraction import dataExtract
from data_extraction import drawMeasurements

#Variable/Parameter Assignment Block
silence = False # silences console output from supported functions.
noise_passes = 5 # denoise passes: decrease to 4 if there are holes, increasing past 7 likely to completely decimate model
extract_thresh = 0.001 # sets the threshold for grabbing points from waist or chest
reconst_depth = 7 # reconstruction depth: amount of subdivisions. definitely exponential complexity, advise going no higher than 10 or 11
mesh_cut = False # experimental floor cutting for the mesh to remove "drape", no time for detailed refinement
paint = True # paint mesh a solid color instead of keeping color information from the point cloud.

# Temporary for testing. Replace with point cloud sent from web
pc = o3d.io.read_point_cloud("point-cloud-example/brandon-5_8_slow.ply")

if(pc.has_points() == False):
    print("There was a problem with the point cloud: Cloud is Empty.")
    exit(1)

pc = cut_floor(pc, silence)

# Had to modify function from original to work
pc = cloud_denoise(pc, noise_passes, silence)

height, wingspan, waistCir, chestCir, height_points, wingspan_points = dataExtract(pc, extract_thresh)

# Replace with function that sends this data to web app
ratio = 68/height
print(
    f"Height: {height} units\n" +
    f"Wingspan: {wingspan} units\n" +
    f"Waist Circumference: {waistCir} units\n" +
    f"Chest Circumference: {chestCir} units\n" +
    f"Actual Height: 68 in\n" +
    f"Wingspan after conversion: {wingspan * ratio} in\n" +
    f"Waist Circumference after conversion: {waistCir * ratio} in\n" +
    f"Chest Circumference after conversion: {chestCir * ratio} in"
)
mesh = generate_mesh(pc, reconst_depth, mesh_cut, paint, silence)

# Draws pc with for testing.
# Comment this out for final build
#drawMeasurements(pc, height, height_points, wingspan_points, waistCloud, everythingelse)
#o3d.visualization.draw_geometries([mesh])
#o3d.visualization.draw_geometries([waistCloud]+waiseLines)
