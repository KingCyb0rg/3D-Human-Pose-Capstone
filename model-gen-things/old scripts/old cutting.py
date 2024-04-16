
#Floor identification procedure (WILL BE REUSED FOR NOISE THAT ISN'T FLOOR)
#Eventually, this will be a function in another file, probably like cull_floor(self, slicecount=50)
slicecount = 50 #easy configuration of slice amount
box = pcd.get_axis_aligned_bounding_box() #Creates bounding box for point cloud

bounds = np.asarray(box.get_box_points()) #Get points for bounding box
bounds = bounds[bounds[:,2].argsort()] #Sort array by z-axis

assembledslices = [] #Assembled slices will be pushed into the array

sliceheight = (box.get_extent()[2]) / slicecount #get arbitrary percentage of height
slices = [box.get_min_bound()[2]] #initialize array for slice boundaries, adding an entry and then using append makes the index always 1 higher than the loop counter

for i in range(slicecount):
    temppcd = o3d.geometry.PointCloud(pcd)
    slices.append(slices[i]+(sliceheight)) #fill slice array with dividers
    assembledslices.append(np.array(bounds)) #create new slice from existing bounds array
    for j in range(4):
        assembledslices[i][j][2] = slices[i] #Change z-values to slice's z-values (bounds array was sorted earlier)
        assembledslices[i][(j+4)][2] = slices[(i+1)] #Forward reference works because array starts with 1 item in it
    assembledslices[i] = temppcd.crop(o3d.geometry.AxisAlignedBoundingBox.create_from_points( #Crop point cloud to bounding box,
                        o3d.utility.Vector3dVector(assembledslices[i]))) #After converting the array to a vector and then to a boundingbox object

    print(assembledslices)
print(assembledslices)