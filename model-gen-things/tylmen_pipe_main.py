import pathlib
import argparse
import open3d as o3d
import numpy as np
import pandas as pd

#Functions for cleaning and measurement
from reconstruct_extract import reconstruct_extract
from tylmen_api_functions import api_master_pipe

def main():
    #argument handling
    parser = argparse.ArgumentParser(
        'Tylmen AR Webapp Backend', description='Backend script for Web app- Uses API to generate Point Cloud, then measures it and creates a 3D Model'
    )
    parser.add_argument('-p', '--passes',
                        help='Amount of noise reduction passes to perform. Defaults to 5. Increasing past 6 will likely result in large holes appearing in the model.',
                        dest='noise_passes',
                        choices=range(1,20),
                        default=5)
    parser.add_argument('-d', '--depth',
                        help='Reconstruction depth. Defaults to 7. higher values will exponentially increase reconstruction computation time (with diminishing returns on increase in fidelity), lower values will reduce model fidelity',
                        dest='recon_depth',
                        choices=range(1,15),
                        default=7)
    parser.add_argument('-t', '--ping-interval',
                        help='Delay in seconds between pings to the server while waiting on generation. Accepted values: 10, 15, 20',
                        choices=[10,15,20],
                        dest='ping_wait',
                        default=15)
    parser.add_argument('-c', '--cut',
                        help='Enables experimental cleaning feature- repeats floor crop after mesh generation. Feet will be missing vertices, not recommended',
                        dest='second_cut_pass',
                        action='store_true')
    parser.add_argument('-n', '--nopaint',
                        help='Disables mesh painting, preserving any color information from the point cloud',
                        dest='paint',
                        action='store_false')
    parser.add_argument('-s', '--hush',
                        help='Disables most console output.',
                        dest='ishushed',
                        action='store_true')
    parser.add_argument('-i', '--infile',
                        help='Input file. Must be .mov',
                        required=True,
                        dest='infile')
    #add -o if it determined to be needed
    args = parser.parse_args()

    #body: opens the provided .mov file and then 
    path = './sandbox/current.ply'
    video_file = open(args.infile, 'rb')
    api_output = api_master_pipe(video_file, hush=args.ishushed, int= args.ping_wait)
    ply = open(path, 'wb')
    ply.write(api_output)
    ply.close()

    #code to output the point cloud file as a .ply, and then take it back in in the open3d format
    point_cloud = o3d.io.read_point_cloud(path)
    mesh, measure_frame = reconstruct_extract(point_cloud, noise_passes=args.noise_passes, mesh_cut=args.second_cut_pass, reconst_depth=args.recon_depth, paint=args.paint, hush=args.ishushed)

    #code to output the mesh and extracted data to a .obj and .csv
    if(args.ishushed == False):
        print("Writing mesh to " + " [directory]")
    o3d.io.write_triangle_mesh('./sandbox/current.obj', mesh)
    if(args.ishushed == False):
        print("Writing measurements to " + " [directory]")
    measure_frame.to_csv('./sandbox/current.csv')


if __name__ == '__main__':
    main()