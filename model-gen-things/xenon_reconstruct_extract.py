import open3d as o3d
import numpy as np
import pandas as pd
import os
import datetime
from timeit import default_timer as time_stamp

#Function for measurement
from data_extraction import dataExtract

#
#   This function runs all of the point cloud processing steps, and inherits the parameters of the called functions.
#
def reconstruct_extract(cloud: o3d.geometry.PointCloud, noise_passes=5, reconst_depth=7, mesh_cut=False, paint=True, hush=False):
    if(hush==False):
        print("Beginning Processing")
    start = time_stamp()
    cut_pcd = cut_floor(cloud, hush)
    denoise_pcd = cloud_denoise(cut_pcd, noise_passes, hush)

    measure_frame, hpt, wpt = dataExtract(denoise_pcd)
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
    stampstring = stamp.isoformat().replace(':', '-').replace('.', '_').replace('T', '_') #replace colons in time with hyphens, other specials with underscores


    fullpath = os.path.join(basedir, stampstring)
    os.makedirs(fullpath)
    return fullpath

#
# Each function below this point supports silent, a bool which disables (most) console output.
#

#
#   Open3d has a tensor module available. If you have GPUs for this to run on, it would probably help, but these functions
#   don't usually take very long on my mid-end i5.
#

#
# cut_floor: takes open3d.geometry.PointCloud, returns same
# finds a plane within the point cloud and crops all points below it. May run into issues with noisy point clouds.
#
def cut_floor(cloud: o3d.geometry.PointCloud, silent=False):
    if(silent == False):
        print("Cropping floor")
        start = time_stamp()
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.003, ransac_n=5, num_iterations=2000)
    #assemble inlier cloud (contains the segmented plane)
    inlier_cloud = cloud.select_by_index(inliers)

    #Get points for bounding box of both inliers and the cloud
    box = cloud.get_axis_aligned_bounding_box()
    inlierbox = inlier_cloud.get_axis_aligned_bounding_box()
    bounds = np.asarray(box.get_box_points())   
    inlierbounds = np.asarray(inlierbox.get_box_points())
    bounds = bounds[bounds[:,2].argsort()] #sorts by z-axis
    inlierbounds = inlierbounds[inlierbounds[:,2].argsort()]

    #Change upper bounds of bounding box to the upper bound of the inlier cloud, and then crop
    for i in range(4):
        bounds[(i+4)][2]= inlierbounds[(i+4)][2]
    #(the open3d functions require certain types, so there's a lot of ugly nesting but it's just type conversions)
    cropped_cloud = cloud.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounds)), invert=True)
    if(silent == False):
        end = time_stamp()
        print("Floor cropped in " + str(end - start) + " seconds")
    return cropped_cloud

#
# cloud_denoise: takes open3d.geometry.PointCloud, returns same
# simple denoise function: defaults to 5 passes, passes after the fifth will repeat an identical pass.
# repeating this pass more than once will likely completely destroy the point cloud.
#
def cloud_denoise(cloud: o3d.geometry.PointCloud, passes = 5, silent=False):
    #if you know of a more readable way to do this please tell me
    if(silent==False):
        start = time_stamp()
        print("Beginning Denoise...")
    if(passes <= 0):
        #will always output to console, unsure how you would end up with this happening (command will not accept 0) but here it is
        print("Denoise started with zero passes, aborting")
    if(passes >= 1):
        if(silent==False):
            print("Initial Count " + str((np.asarray(cloud.points).size)/3) + " Points")
        #these functions remove points with useless information in them (duplicate positions/infinite or nan coordinates)
            print("Removing Bad/Dupe points before cleaning")
        prepared_cloud = cloud.remove_duplicated_points()
        prepared_cloud = prepared_cloud.remove_non_finite_points()
        if(silent==False):
            print("New Count " + str((np.asarray(prepared_cloud.points).size)/3) + " Points, beginning passes")
        denoise_cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.2) #std_ratio is the standard deviation a point must be 'OUTSIDE of to BE REMOVED', not 'inside of to avoid removal'
        if(silent==False):
            p1 = time_stamp()
            print("Pass 1: " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(p1 - start) + " seconds")
    if(passes >= 2):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.7)
        if(silent==False):
            p2 = time_stamp()
            print("Pass 2: " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(p2 - p1) + " seconds")
    if(passes >= 3):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.8)
        if(silent==False):
            p3 = time_stamp()
            print("Pass 3: " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(p3 - p2) + " seconds")
    if(passes >= 4):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.4)
        if(silent==False):
            p4 = time_stamp()
            print("Pass 4: " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(p4 - p3) + " seconds")
    if(passes >= 5):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.65)
        if(silent==False):
            p5 = time_stamp()
            print("Pass 5: " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(p5 - p4) + " seconds")

            #The above passes:
            #1: low requirement, strict filter
            #2: moderate requirement, lenient
            #3: huge requirement, lenient filter
            #4: moderate requirement, more strict
            #5: large requirement, somewhat lenient
            #remove_statistical_outlier works by comparing the relative distances between points and their neighbors. ones that are too far away (comparatively)
            #are discarded.

            #There is another function that could be useful if this code is actually used, remove_radius_outlier
            #Whenever I tried to call it, it gave a typeerror despite the arguments being correct (I checked many, many times)
            #If you can get it to work, it could be useful to replace some of these statistical functions with the radius one
            #(because of the inconsistent scaling, I'd recommend you have it work based off a percentage of the height of the point cloud
            #instead of hardcoding a radius. obtaining this height in open3d is very easy, use get_axis_aligned_bounding_box(), get_max_bound(),
            # and get_min_bound(). the bound functions return a point as a 1x3 numpy array, so the height value can be accessed with get_x_bound()[2].)

    if(passes > 5):
        if(silent==False):
            pprevious = p5
        for x in range(passes - 5):
            denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.8) #repeatable medium filter
            #going past 7 passes is very likely to completely decimate the model, 5 was already pushing it in some test cases. adjust to needs
            if(silent==False):
                ploop = time_stamp()
                print("Pass " + str(x+5) +": " + str((np.asarray(denoise_cloud.points).size)/3) + " Points in " + str(ploop - pprevious) + " seconds")
                pprevious = ploop

    if(passes > 0):
        if(silent==False):
            end = time_stamp()
            print("Denoise complete in " + str(end - start) + " seconds")
        return denoise_cloud
    else:
        return cloud #case of zero passes

#
# generate_mesh: takes open3d.geometry.PointCloud, returns open3d.geometry.TriangleMesh
# reconstructs a model from a provided point cloud. optional cleaning step included, off by default. removes extra trimmings from
# the bottom of the mesh, but it cuts into existing triangles. depth affects level of detail in mesh.
# paint_mesh removes color information from the generated mesh.
#
def generate_mesh(cloud: o3d.geometry.PointCloud, depth = 7, experimental_clean=False, paint_mesh=True, silent=False):
    #measurement necessary for experimental clean
    cloud_low_bound = cloud.get_min_bound()[2]
    if(silent==False):
        start = time_stamp()
        print("Generating model with depth " + str(depth)) #not that this takes time before depth 9, but still

    #Reconstruction: algorithm requires normals
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    d = depth
    reconstructed_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=d, linear_fit=True)
    if(silent==False):
        end = time_stamp()
        print("Generation complete in " + str(end-start) + " seconds") 

    if(paint_mesh==True):
        reconstructed_mesh.paint_uniform_color([0.1,0.7,0.5])
        if(silent==False):
            print("Mesh Painted")

    if(experimental_clean == True):
        #measurement from earlier crops points in the exact same manner as cut_floor, feet will be jagged. off by default
        meshbounds = np.asarray(reconstructed_mesh.get_axis_aligned_bounding_box().get_box_points())
        meshbounds = meshbounds[meshbounds[:,2].argsort()]
        for i in range(4):
            meshbounds[i][2] = cloud_low_bound
        cropped_mesh = reconstructed_mesh.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(meshbounds)))
        return cropped_mesh
    else:
        return reconstructed_mesh

#
# cloud_slice: takes PointCloud and int, returns list of PointCloud with (int) items
# Slices a point cloud into a configurable number of pieces along the z-axis.
# unused, included for completeness. would have been more applicable to measurement as a brute-force
# solution to certain problems, but other approximation methods were used.
#
def cloud_slice(cloud: o3d.geometry.PointCloud, slices = 50, silent = False):
    #Obtain bounding box of full cloud
    box = cloud.get_axis_aligned_bounding_box()
    bounds = np.asarray(box.get_box_points())
    bounds = bounds[bounds[:,2].argsort()] #Sort points by z-axis

    builtslices = []
    sliceheight = (box.get_extent()[2]) / slices #divide by configurable value
    sliceboundarray = [box.get_min_bound()[2]] #initialize array for storage of height dividers with one value
    if(silent == False):
        start = time_stamp()
        print("Slicing into " + slices + " parts")
    for i in range(slices):
        temppcd = o3d.geometry.PointCloud(cloud) #copy constructor for no reason
        sliceboundarray.append(sliceboundarray[i]+(sliceheight)) #fill slice array with dividers
        builtslices.append(np.array(bounds)) #create new slice from existing bounds array
        for j in range(4):
            builtslices[i][j][2] = sliceboundarray[i] #Change z-values to slice's z-values (bounds array was sorted earlier)
            builtslices[i][(j+4)][2] = sliceboundarray[(i+1)] #the array starts with one item in it so this does not result in an error
            #crop with bounding box and push to array
            builtslices[i] = temppcd.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(builtslices[i]))) 
    if(silent == False):
        end = time_stamp()
        print("Slice completed in " + str(end-start) + " seconds")
    return builtslices

