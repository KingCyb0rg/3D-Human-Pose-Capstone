import open3d as o3d
import numpy as np
import math

# Input Pointcloud object  Output: Returns height and width float, topPoint and bottomPoint list, leftPoint and rightPoint list
# Extracts height and width from a pointcloud
def dataExtract(pointcloud, threshold=0.001):

    # Input: Pointcloud object  Output: top and bottom cooridnates float, and height float
    def getHeight(pointcloud):
        vertices =  np.asarray(pointcloud.points)

        # Sorts by y axis coordinate
        sorted_vertices = vertices[vertices[:, 1].argsort()]
        
        height = sorted_vertices[-1][1] - sorted_vertices[0][1]
        return sorted_vertices[-1].tolist(), sorted_vertices[0].tolist(), height

    # Input: Pointcloud object  Output: top and bottom cooridnates float, and height float
    def getWingSpan(pointcloud):
        vertices =  np.asarray(pointcloud.points)

        # Sorts by x axis coordinate
        sorted_vertices_x = vertices[vertices[:, 0].argsort()]
        width_x = sorted_vertices_x[-1][0] - sorted_vertices_x[0][0]
        # Sorts by z axis coordinate
        sorted_vertices_z = vertices[vertices[:, 2].argsort()]
        width_z = sorted_vertices_z[-1][2] - sorted_vertices_z[0][2]

        if (width_x > width_z):
            width = width_x
            sorted_vertices = sorted_vertices_x
        else:
            width = width_z
            sorted_vertices = sorted_vertices_z

        return sorted_vertices[-1].tolist(), sorted_vertices[0].tolist(), width
    
    def getWaist(pointcloud, height, threshold):
        vertices =  np.asarray(pointcloud.points)

        # Sorts by y axis coordinate
        sorted_vertices = vertices[vertices[:, 1].argsort()]
        topPoint = sorted_vertices[-1]

        # Using 0.45 to approximate where the waist is.
        waistPoint = topPoint.copy()
        waistPoint[1] = waistPoint[1] - (height * 0.45)

        y_max = waistPoint[1] + threshold
        y_min = waistPoint[1] - threshold

        # Grabs the indices that represents the waist within threshold
        waist_indices = []
        pointIsCovered = {}
        for index, v in enumerate(vertices):
            if v[1] < y_max and v[1] > y_min:
                v_key = (v[0], v[1], v[2])

                waist_indices.append(index)
                pointIsCovered[v_key] = False
        
        # Used to visualize the points being grabbed and separate waist from point cloud
        waist_cloud = pointcloud.select_by_index(waist_indices)
        everythingelse = pointcloud.select_by_index(waist_indices, invert=True)
        waist_cloud.paint_uniform_color([0, 1, 0])

        # Finds center point of waist.
        waist_vertices = np.asarray(waist_cloud.points)
        waist_center_point = np.asarray([
            np.average(waist_vertices[:,0]),
            np.average(waist_vertices[:,1]),
            np.average(waist_vertices[:,2])
            ])

        # Sorts
        angles = np.arctan2(waist_vertices[:, 0] - waist_center_point[0], waist_vertices[:, 2] - waist_center_point[2])
        indices = np.argsort(angles)
        sorted_waist_vertices = waist_vertices[indices]

        # Calculates perimeter of waist.
        sum = 0
        lines = []
        for i, v1 in enumerate(sorted_waist_vertices):
            if i+1 < len(sorted_waist_vertices):
                v2 = sorted_waist_vertices[i+1]
            
            distance = math.sqrt( pow(v2[0] - v1[0], 2) + pow(v2[2] - v1[2], 2) )
            print(distance)
            sum += distance

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([v2, v1]),
                lines=o3d.utility.Vector2iVector([[0,1]]),
            )
            line_set.paint_uniform_color((0, 1, 0))
            lines.append(line_set)

        print(sum)

        return waist_cloud

    def getChest(pointcloud, height, threshold):
        vertices =  np.asarray(pointcloud.points)
        div = height/32

        # Sorts by y axis coordinate
        sorted_vertices = vertices[vertices[:, 1].argsort()]
        topPoint = sorted_vertices[-1]

        # Using 0.45 to approximate where the waist is.
        chestPoint = topPoint.copy()
        chestPoint[1] = chestPoint[1] - (height - 23 * div)

        y_max = chestPoint[1] + threshold
        y_min = chestPoint[1] - threshold

         # Grabs the indices that represents the chest within threshold
        chest_indices = []
        pointIsCovered = {}
        for index, v in enumerate(vertices):
            if v[1] < y_max and v[1] > y_min:
                v_key = (v[0], v[1], v[2])

                chest_indices.append(index)
                pointIsCovered[v_key] = False

        # Used to visualize the points being grabbed and separate chest from point cloud
        chest_cloud = pointcloud.select_by_index(chest_indices)
        everythingelse = pointcloud.select_by_index(chest_indices, invert=True)
        chest_cloud.paint_uniform_color([1, 0, 1])

        return chest_cloud, everythingelse
        
    
    # Rotates point cloud to correct orientation for measurements
    # May cause program to break if point cloud generation methods change
    rotation = pointcloud.get_rotation_matrix_from_xyz((-np.pi / 2, 0, -np.pi / 2))
    pointcloud.rotate(rotation)

    
    
    topPoint, bottomPoint, height = getHeight(pointcloud)
    leftPoint, rightPoint, wingspan = getWingSpan(pointcloud)
    waistCloud = getWaist(pointcloud, height, threshold)
    chestCloud, everythingelse = getChest(pointcloud, height, threshold)

    threshold = height/threshold

    #waistPoint1, waistPoint2, waistWidth = getWaist(pointcloud, leftPoint, threshold)

    return height, wingspan, [topPoint, bottomPoint], [leftPoint, rightPoint], chestCloud, everythingelse


def drawMeasurements(pointcloud, height, height_points, width_points, waistCloud, everythingelse):

    # Top sphere to show topmost point
    sphereTop = o3d.geometry.TriangleMesh.create_sphere().translate(height_points[0], relative = False)
    sphereTop.scale(0.02, center = sphereTop.get_center())
    sphereTop.paint_uniform_color((0, 1, 0))

    # Bottom sphere to show bottommost point
    sphereBottom = o3d.geometry.TriangleMesh.create_sphere().translate(height_points[1], relative = False)
    sphereBottom.scale(0.02, center = sphereBottom.get_center())
    sphereBottom.paint_uniform_color((0, 1, 0))

    # Gets line height without x or z coordinates to make a vertical line.
    line_set_height = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(height_points),
        lines=o3d.utility.Vector2iVector([[0,1]]),
    )
    line_set_height.paint_uniform_color((0, 1, 0))


    # Left sphere to show leftmost point
    sphereLeft = o3d.geometry.TriangleMesh.create_sphere().translate(width_points[0], relative = False)
    sphereLeft.scale(0.02, center = sphereLeft.get_center())
    sphereLeft.paint_uniform_color((1, 0, 0))

    # Bottom sphere to show bottommost point
    sphereRight = o3d.geometry.TriangleMesh.create_sphere().translate(width_points[1], relative = False)
    sphereRight.scale(0.02, center = sphereRight.get_center())
    sphereRight.paint_uniform_color((1, 0, 0))

    # Gets line height without x or z coordinates to make a vertical line.
    line_set_width = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(width_points),
        lines=o3d.utility.Vector2iVector([[0,1]]),
    )
    line_set_width.paint_uniform_color((1, 0, 0))

    # Waist sphere to show waist point
    waistPoint = height_points[0].copy()
    waistPoint[1] = waistPoint[1] - (height * 0.45)
    sphereWaist = o3d.geometry.TriangleMesh.create_sphere().translate(waistPoint, relative = False)
    sphereWaist.scale(0.02, center = sphereWaist.get_center())
    sphereWaist.paint_uniform_color((0, 1, 0))

    ball = o3d.geometry.TriangleMesh.create_sphere().translate((0,0,0), relative = False)
    ball.scale(0.02, center = ball.get_center())

    #o3d.visualization.draw_geometries([pointcloud, sphereTop, sphereBottom, line_set_height, sphereLeft, sphereRight, line_set_width])
    o3d.visualization.draw_geometries([sphereTop, sphereBottom, sphereRight, sphereLeft, line_set_height, line_set_width, waistCloud, everythingelse, pointcloud, ball])