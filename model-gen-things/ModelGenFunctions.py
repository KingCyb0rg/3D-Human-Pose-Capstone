import open3d as o3d
import numpy as np

#
# cut_floor: takes open3d.geometry.PointCloud, returns same
# finds a plane within the point cloud and crops all points below it. May run into issues with noisy point clouds.
#
def cut_floor(cloud):
    print("Cropping Floor")
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.003,
                                             ransac_n=5,
                                             num_iterations=2000)
    #assemble inlier cloud (contains the segmented plane)
    inlier_cloud = cloud.select_by_index(inliers)
    #Get points for bounding box of both inliers and the cloud
    box = cloud.get_axis_aligned_bounding_box()
    inlierbox = inlier_cloud.get_axis_aligned_bounding_box()
    bounds = np.asarray(box.get_box_points())   
    inlierbounds = np.asarray(inlierbox.get_box_points())
    bounds = bounds[bounds[:,2].argsort()]
    inlierbounds = inlierbounds[inlierbounds[:,2].argsort()]
    #Change bounds of inlier cloud to the lower bound of the main cloud, and then crop
    for i in range(4):
        bounds[(i+4)][2]= inlierbounds[(i+4)][2]
    cropped_cloud = cloud.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounds)), invert=True)
    print("Floor Cropped")
    return cropped_cloud

#
# cloud_denoise: takes open3d.geometry.PointCloud and int, returns PointCloud (or 0 in the case of passes being zero or negative
# simple denoise function: defaults to 5 passes, passes after the fifth will repeat an identical pass.
# repeating this pass more than once will likely completely destroy the point cloud.
#
def cloud_denoise(cloud, passes = 5):
    print("Beginning Denoise...")
    if(passes <= 0):
        print("Denoise started with zero passes, aborting")
    if(passes >= 1):
        print("Initial Count " + (np.asarray(cloud.points).size)/3 + " Points")
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=15,
                                                        std_ratio=1.9)
        print("Pass 1: " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes >= 2):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.7)
        print("Pass 2: " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes >= 3):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=25,
                                                        std_ratio=1.6)
        print("Pass 3: " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes >= 4):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=100,
                                                        std_ratio=1.2)
        print("Pass 4: " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes >= 5):
        denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=85,
                                                        std_ratio=1.3)
        print("Pass 5: " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes > 5):
        for x in range(passes - 5):
            denoise_cloud, ind = denoise_cloud.remove_statistical_outlier(nb_neighbors=30,
                                                            std_ratio=1.5)
            print("Pass " + x +": " + (np.asarray(denoise_cloud.points).size)/3 + " Points")
    if(passes < 0):
        print("Denoise Complete")
        return denoise_cloud
    else:
        return 0

#
# generate_mesh: takes open3d.geometry.PointCloud, an integer, and a bool, returns open3d.geometry.TriangleMesh
# reconstructs a model from a provided point cloud. optional cleaning step included, off by default. removes extra trimmings from
# the bottom of the mesh, but it cuts into existing triangles. depth affects level of detail in mesh.
#
def generate_mesh(cloud, depth = 7, experimental_clean=False):
    cloud_low_bound = cloud.get_min_bound()[2]
    #Reconstruction: algorithm requires normals
    print("Generating Model...")
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    d = 7
    reconstructed_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=d, linear_fit=True)
    print("Generation Complete") 
    if(experimental_clean == True):
        meshbounds = np.asarray(reconstructed_mesh.get_axis_aligned_bounding_box().get_box_points())
        meshbounds = meshbounds[meshbounds[:,2].argsort()]
        for i in range(4):
            meshbounds[i][2] = cloud_low_bound
        cropped_mesh = reconstructed_mesh.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(meshbounds)))
        return cropped_mesh
    else:
        return reconstructed_mesh




