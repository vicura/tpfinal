#archivo para clasificar simulaciones con PointNet

import sys
import os
import argparse
import numpy as np
import MDAnalysis as mda
import networkx as nx

# For PointNet modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

import nsgrid_rsd as nsgrid
from pointnet import PointNet

def main():

    #Parse Arguments
    parser = argparse.ArgumentParser(description='Uses MDAnalysis and PointNet to identify largest cluster of solid-like atoms')
    parser.add_argument('--weights', help='folder containing network weights to use', type=str, required=True)
    parser.add_argument('--nclass', help='number of classes', type=int, required=True)
    parser.add_argument('--trjpath', help='path to files', type=str, required=True)
    parser.add_argument('--cutoff', help='neighbor cutoff distance (in nm)', type=float, required=True)
    parser.add_argument('--maxneigh', help='max number of neighbors', type=int, required=True)
    parser.add_argument('--outname', help='output file name', type=str, required=True)

    args = parser.parse_args()

    # Import topology
    u = mda.Universe(args.trjpath,topology_format='LAMMPSDUMP')
    
    # File to write output
    f_summary = open(args.outname+'_summary.mda','w')
    f_class = open(args.outname+'_class.mda','w')
    
    f_summary.write("# Time, n_lam, n_hex\n")
    f_class.write("# Frame, Átomo, Resultado, x, y, z\n")

    # Here is where we initialize the pointnet
    pointnet = PointNet(n_points=args.maxneigh,n_classes=args.nclass,weights_dir=args.weights)

    # Loop over trajectory
    for ts in u.trajectory:
        # Generate neighbor list (dentro de las coordenadas especificadas)
        print("Generating neighbor list") 
        nlist = nsgrid.FastNS(args.cutoff*1.0,u.atoms.positions,ts.dimensions).self_search()

        # Extract required info 
        ndxs = nlist.get_indices()
        dxs = nlist.get_dx()
        dists = nlist.get_distances()
        print("Extracted all relevant information") 

        samples = []
        # Prepare samples to send through pointnet
        for i in range(len(dxs)):
            ## Sort neighbors by distance (so that we can 
            ## normalize all distances such that the distance
            ## to the closest neighbor is 1.0 )
            nneigh = int(len(dxs[i])/3)
            np_dxs = np.asarray(dxs[i]).reshape([nneigh,3])
            sort_order = np.asarray(dists[i]).argsort()
            # Sort neighbors by distance 
            np_dxs = np_dxs[sort_order]
            if nneigh > 0:
                np_dxs /= np.linalg.norm(np_dxs[0])
            # Now correctly size/pad the point cloud
            if nneigh < args.maxneigh:
                np_dxs = np.pad(np_dxs,[(0, args.maxneigh-nneigh), (0, 0)],'constant',)
            elif nneigh > args.maxneigh:
                np_dxs = np_dxs[:args.maxneigh]

            # Append sample info
            samples.append(np_dxs)

        # And convert to np array
        np_samples = np.asarray(samples)
        print("Frame {}, Shape sent to pointnet: {}".format(ts.frame,np_samples.shape))
        sys.stdout.flush()
    
        # Send sample through inference (se hacen las predicciones)
        results = pointnet.infer_nolabel(np_samples)
        results = np.asarray(results)
    
        print("Frame {}, Results returned from pointnet, shape {}".format(ts.frame,results.shape))
        sys.stdout.flush()

        # Extract different atom types
        lam_atoms = np.where(results == 0)[0]
        hex_atoms = np.where(results == 1)[0]



        
        f_summary.write("{:8.3f}{:8d}{:8d}\n".format(ts.time,lam_atoms.shape[0],hex_atoms.shape[0]))
        
        for atom in u.atoms:
            f_class.write("{:10d}{:8d}{:8d}{:^20.10f}{:^20.10f}{:^20.10f}\n".format(ts.frame,atom,results[atom.index],atom.position[1],atom.position[2],atom.position[3]))  #indica a que clase pertenece cada nodo del 
                                                                                       #largest cluster

    f_summary.close()
    f_class.close()

# Boilerplate command to exec main
if __name__ == "__main__":
    main()
