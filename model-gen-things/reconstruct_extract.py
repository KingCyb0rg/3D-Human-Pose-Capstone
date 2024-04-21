import open3d as o3d
import numpy as np
import pandas as pd
import os
import datetime
from timeit import default_timer as time_stamp

#Functions for cleaning and measurement
from model_gen_functions import cut_floor
from model_gen_functions import cloud_denoise
from model_gen_functions import generate_mesh
from data_extraction import dataExtract

#
#   This function runs all of the point cloud processing steps.
#
def reconstruct_extract(cloud: o3d.geometry.PointCloud, noise_passes=5, waist_thresh=8, reconst_depth=7, mesh_cut=False, paint=True, hush=False):
    if(hush==False):
        print("Beginning Processing")
    start = time_stamp()
    cut_pcd = cut_floor(cloud, hush)
    denoise_pcd = cloud_denoise(cut_pcd, noise_passes, hush)

    measure_frame, hpt, wpt = dataExtract(denoise_pcd, waist_thresh)
    if(hush==False):
        print("Measurements created")
    poisson_mesh = generate_mesh(denoise_pcd, reconst_depth, mesh_cut, paint, hush)
    end = time_stamp()
    if(hush==False):
        print("Processing completed in " + str(end-start) + " seconds")
    return poisson_mesh, measure_frame

#
#   This function creates and returns the path to place output files into:
#   tylmen-xenon-out/[datestamp][timestamp]/pointcloud.ply
#   tylmen-xenon-out/[datestamp][timestamp]/mesh.obj
#   tylmen-xenon-out/[datestamp][timestamp]/measurements.csv
#   it's called xenon because freon is trademarked
#
def build_path():
    basedir = os.path.join(".", "tylmen-xenon-out")
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    #name folders by timestamp
    stamp = datetime.datetime.now()
    stamp.microsecond = 0 #setting microns to 0 to remove them from iso format output
    stampstring = stamp.isoformat()

    fullpath = os.path.join(basedir, stampstring)
    os.makedirs(fullpath)
    return fullpath