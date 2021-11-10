# Ryan DeFever
# Sarupria Research Group
# Clemson University
# 2019 Jun 10

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
    dos = u.select_atoms('type 2')
    
    # File to write output
    f_summary = open(args.outname+'_summary.mda','w')
    f_class = open(args.outname+'_class.mda','w')
    
    f_summary.write("{:^8s}{:^25s}{:^8s}{:^8s}\n".format('Time','Tamaño del Largest_cls','n_lam','n_hex'))
    f_class.write("{:^10s}{:^8s}{:^8s}{:^20s}{:^20s}{:^20s}\n".format('Frame','Nodo','Resultado','x','y','z'))

    # Here is where we initialize the pointnet
    pointnet = PointNet(n_points=args.maxneigh,n_classes=args.nclass,weights_dir=args.weights)

    # Loop over trajectory
    for ts in u.trajectory:
        # Generate neighbor list (dentro de las coordenadas especificadas)
        print("Generating neighbor list") 
        nlist = nsgrid.FastNS(args.cutoff*10.0,dos.positions,ts.dimensions).self_search()

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
        #other_atoms = np.where(results > 1)[0]
        #print("%d total other atoms" % other_atoms.shape[0])

        ## Now we are going to construct the largest cluster of
        ## solid atoms in the system (i.e., a solid nucleus)

        # We need neighbor lists for connectivity cutoff 
        # Using 5.0 Angstroms (mda units) here
        nlist = nsgrid.FastNS(5.0,dos.positions,ts.dimensions).self_search()   #que cutoff usar??

        pairs = nlist.get_pairs()   #get_pairs returns all the pairs within the desired cutoff distance

        # Find the largest cluster of solids 
        G = nx.Graph()					#Create an empty graph structure (a “null graph”) with no nodes and no edges.
        G.add_edges_from(pairs)                       #G is grown adding a list of edges (en este caso los edges se forman al listar los 
                                                      # pares de vecinos considerados dentro del cutoff indicado)
        #G.remove_nodes_from(liquid_atoms)            # aca se sacaron los nodos liquidos porque buscan e cluster de atomos sólidos más 
                                                      # grande (?. No nos serviria a nosotros?
        largest_cluster = G.subgraph(max(nx.connected_components(G), key=len))     # Subgraph view of the graph, consists in the  
                                                                                   # largest connected component of the graph (mayor
                                                                                   # cantidad de vecinos conectados?)
        
        pos = nx.spring_layout(largest_cluster, dim=3)            # coordenadas de los nodos del largest cluster
        
        f_summary.write("{:8.3f}{:25d}{:8d}{:8d}\n".format(ts.time,len(largest_cluster),lam_atoms.shape[0],
           hex_atoms.shape[0]))
        
        for node in largest_cluster:
            f_class.write("{:^10d}{:^8d}{:^8d}{:^20.10f}{:^20.10f}{:^20.10f}\n".format(ts.frame+1,node,results[node],pos[node][0],pos[node][1],pos[node][2]))  #indica a que clase pertenece cada nodo 
                                                                                                 # del largest cluster

    f_summary.close()
    f_class.close()

# Boilerplate command to exec main
if __name__ == "__main__":
    main()
