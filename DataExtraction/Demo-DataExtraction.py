import open3d as o3d
import numpy as np

from data_extraction import dataExtract
from data_extraction import drawMeasurements

def removeOutliers(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003,
                                            ransac_n=5,
                                            num_iterations=2000)

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    #box = inlier_cloud.get_axis_aligned_bounding_box() #Creates bounding box for point cloud
    #bounds = np.asarray(box.get_box_points()) #Get points for bounding box
    #bounds = bounds[bounds[:,2].argsort()]
    #for i in range(4):
    #    bounds[i][2] = pcd.get_min_bound()[2]
    #cpcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounds)), invert=True)

    box = pcd.get_axis_aligned_bounding_box() #Creates bounding box for point cloud
    inlierbox = inlier_cloud.get_axis_aligned_bounding_box()
    bounds = np.asarray(box.get_box_points()) #Get points for bounding box
    inlierbounds = np.asarray(inlierbox.get_box_points())
    bounds = bounds[bounds[:,2].argsort()]
    inlierbounds = inlierbounds[inlierbounds[:,2].argsort()]

    for i in range(4):
        bounds[(i+4)][2]= inlierbounds[(i+4)][2]
    cpcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounds)), invert=True)

    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=15,
                                                        std_ratio=2.0)

    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.7)

    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=25,
                                                        std_ratio=1.6)

    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=100,
                                                        std_ratio=1.3)
    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=85,
                                                        std_ratio=1.6)
    cpcd, ind = cpcd.remove_statistical_outlier(nb_neighbors=85,
                                                        std_ratio=1.4)

    cloud_low_bound = cpcd.get_min_bound()[2]

    return cpcd

pc = o3d.io.read_point_cloud("point-cloud-example/brandon-5_8_slow.ply")
cleaned_pc = removeOutliers(pc)

height, width, height_points, width_points = dataExtract(cleaned_pc)

drawMeasurements(cleaned_pc, height, width, height_points, width_points)