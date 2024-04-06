import open3d as o3d
import numpy as np

# Input Pointcloud object  Output: Returns height and width float, topPoint and bottomPoint list, leftPoint and rightPoint list
# Extracts height and width from a pointcloud
def dataExtract(pointcloud, threshold=8):

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
    
    def getWaist(pointcloud, ylimit_point, threshold):
        # AHH
        return
    
    # Rotates point cloud to correct orientation for measurements
    # May cause program to break if point cloud generation methods change
    rotation = pointcloud.get_rotation_matrix_from_xyz((-np.pi / 2, 0, -np.pi / 2))
    pointcloud.rotate(rotation)

    
    
    topPoint, bottomPoint, height = getHeight(pointcloud)
    leftPoint, rightPoint, wingspan = getWingSpan(pointcloud)

    threshold = height/threshold

    #waistPoint1, waistPoint2, waistWidth = getWaist(pointcloud, leftPoint, threshold)

    return height, wingspan, [topPoint, bottomPoint], [leftPoint, rightPoint]


def drawMeasurements(pointcloud, height_points, width_points):

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

    o3d.visualization.draw_geometries([pointcloud, sphereTop, sphereBottom, line_set_height, sphereLeft, sphereRight, line_set_width])