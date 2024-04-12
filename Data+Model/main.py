import open3d as o3d
import numpy as np

from data_extraction import *
from ModelGenFunctions import *

# Temporary for testing. Replace with point cloud sent from web
pc = o3d.io.read_point_cloud("point-cloud-example/brandon-5_8_slow.ply")

pc = cut_floor(pc)

# Had to modify function from original to work
pc = cloud_denoise(pc)

height, wingspan, height_points, wingspan_points, waistCloud, everythingelse = dataExtract(pc)

# Replace with function that sends this data to web app
ratio = 68/height
print(
    f"Height: {height} units\n" +
    f"Wingspan: {wingspan} units\n" +
    f"Actual Height: 68 in\n" +
    f"Wingspan after conversion: {wingspan * ratio} in"
)
mesh = generate_mesh(pc)

# Draws pc with measurement to see if everything is correct.
# Comment this out for final build
drawMeasurements(pc, height, height_points, wingspan_points, waistCloud, everythingelse)
#o3d.visualization.draw_geometries([mesh])
