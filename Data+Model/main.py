import open3d as o3d
import numpy as np

from data_extraction import *
from ModelGenFunctions import *

pc = o3d.io.read_point_cloud("point-cloud-example/brandon-5_8_slow.ply")

pc = cut_floor(pc)

# Had to modify function from original to work
pc = cloud_denoise(pc)

height, wingspan, height_points, wingspan_points = dataExtract(pc)

# Draws pc with measurement to see if everything is correct.
# Comment this out for final build
print(
    f"Height: {height} units\n" +
    f"Wingspan: {wingspan} units"
)
drawMeasurements(pc, height_points, wingspan_points)