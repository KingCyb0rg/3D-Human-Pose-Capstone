import open3d as o3d
import numpy as np
import pandas as pd

#Functions for cleaning and measurement
from model_gen_functions import cut_floor
from model_gen_functions import cloud_denoise
from model_gen_functions import generate_mesh
from data_extraction import dataExtract

def reconstruct_extract(cloud: o3d.geometry.PointCloud, noise_passes=5, waist_thresh=8, reconst_depth=7, mesh_cut=False, paint=True, hush=False):

    cut_pcd = cut_floor(cloud, hush)
    denoise_pcd = cloud_denoise(cut_pcd, noise_passes, hush)

    measure_frame = dataExtract(denoise_pcd, waist_thresh)
    poisson_mesh = generate_mesh(denoise_pcd, reconst_depth, mesh_cut, paint, hush)

    return poisson_mesh, measure_frame



    