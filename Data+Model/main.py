import open3d as o3d
import numpy as np

from data_extraction import *
from ModelGenFunctions import *

pc = o3d.io.read_point_cloud("point-cloud-example/brandon-5_8_slow.ply")

pc = cut_floor(pc)
pc = cloud_denoise(pc)

height, width, height_points, width_points = dataExtract(pc)
drawMeasurements(pc, height, width, height_points, width_points)