import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import MDAnalysis as mda
import argparse
import multiprocessing
from multiprocessing import Pool
import time
import nsgrid_rsd as nsgrid

"""Para que la red reconozca cada mesofase que adopta el sistema, le doy en el entrenamiento varias muestras de cada una: lamelar desordenada, lamelar y lamelar ordenada.


Estas muestras consisten en outputs .lammpstj de corridas en condiciones constantes de presión y temperatura (NPT). Para cada mesofase tengo 4 outputs correspondientes a 4 temperaturas distintas pero cercanas.

A continuación, con la función pre_proc, se recopilan las muestras, se las divide de acuerdo a la mesofase a la que corresponden"""

# Commented out IPython magic to ensure Python compatibility.
#Función modificada de https://github.com/rsdefever/GenStrIde/blob/master/scripts/prepare_train.py

def main():

    args = get_args()
    # Lista de clases

    classes = ['lam','iso']
    
    # Número de clases

    nclass = len(classes)

    # Construyo un único archivo para cada mesofase que contiene los
    # .lammpstj correspondientes
    list_dir = os.listdir(args.path)
    iso = [d for d in list_dir if d.startswith('dump_iso')]
    
    with open('dump.iso', 'w') as outfile:
       for fname in iso:
          with open(fname) as infile:
             for line in infile:
                outfile.write(line)
    
    lam = [d for d in list_dir if d.startswith('dump_lam')]       
 
    with open('dump.lam', 'w') as outfile:
       for fname in lam:
          with open(fname) as infile:
             for line in infile:
                outfile.write(line)
                
    
    files = ['dump.lam','dump.iso']          


    # Divido en samples y labels
    
    samples = []
    labels = []


    for fcount in range(len(files)):
       print(files[fcount])
    # Extraigo la class id y creo label

       if 'dump.iso' in files[fcount]:
          classid = 'iso' 
       elif 'dump.lam' in files[fcount]:
          classid = 'lam'

       label = np.zeros(len(classes))   
       ndx = classes.index(classid)
       label[ndx] = 1
   
       # Leo los archivos con MDAnalysis

       u = mda.Universe(files[fcount],topology_format='LAMMPSDUMP')
       print('numero de frames: ',u.trajectory.n_frames)
       print('numero de atomos totales: ',u.trajectory.n_atoms)
       # Para cada frame de la trayectoria:
       for ts in u.trajectory:

          # Selecciono aleatoriamente n_select átomos 
          sel = np.random.choice(u.atoms.n_atoms,size=args.n_select,replace=False)
      
          # Creo una lista de vecinos para los átomos seleccionados
          nlist = nsgrid.FastNS(args.cutoff*1.0,u.atoms.positions,ts.dimensions).search(u.atoms[sel].positions)
          ndxs = nlist.get_indices()             # Devuelve los indices de los 
                                                 # vecinos al átomo
          dxs = nlist.get_dx()                   # Devuelve coordenadas de los 
                                                 # vecinos
          dists = nlist.get_distances()          # Devuelve las distancias 
                                                 # individiales de cada uno de 
                                                 # los vecinos
          # Itero sobre los átomos seleccionados
          for i in range(len(sel)):
            
             np_dxs = np.asarray(dxs[i]).reshape(-1,3)
             
             sort = np.argsort(dists[i])         # Ordeno de acuerdo a las 
                                                 # distancias
             vals = np_dxs[sort][1:]             # Remuevo el átomo (0,0,0) 
                                                 # (que es el átomo seleccionado 
                                                 # para buscar sus vecinos)
             nneigh = vals.shape[0]              # Número de vecinos


             # Zero padding 
             # Si la muestra contiene más de max_neigh partículas, se le aplica 
             # resize
             # Si la muestra contiene menos de max_neigh partículas, la pointcloud
             # se rellena con puntos (0, 0, 0) hasta llegar a max_neigh
             if nneigh > args.max_neigh:
                sample = np.resize(vals,(args.max_neigh,3))
             elif nneigh < args.max_neigh:
                npad = args.max_neigh  - nneigh
                sample = np.vstack((vals,np.zeros((npad,3))))
             else:
                sample = vals

             samples.append(sample)   #agrego a samples
             labels.append(label)     #agrego a labels

    # Convierto samples y labels en arrays 
    samples = np.asarray(samples)
    print('samples shape', samples.shape)
    labels = np.asarray(labels)
    print('labels shape', labels.shape)

    ## Extra pre-processing de los datos de entrenamiento ##
    # Extraigo idxs para las diferentes clases
    idxs = [np.where(labels[:,i] == 1)[0] for i in range(nclass)]
    # Extraigo samples/labels para cada clase
    samples_list = [samples[idx] for idx in idxs]
    labels_list = [labels[idx] for idx in idxs]
    # 1. Hago shuffle dentro de cada clase -- returns [None,none,none,none]
    [np.random.shuffle(samples_list[i]) for i in range(nclass)]
    # 2. Confirmo igual número de muestras en cada clase
    for i in range(nclass):
        print("Total number of available samples for class %s: %d" % (classes[i],
                                                                      samples_list[i].shape[0]))
        assert samples_list[i].shape[0] >= args.n_samples, \
            "Error, only %d samples in class %s but requested %d." \
#             % (samples_list[i].shape[0],classes[i],n_samples)
    samples_list = [samples_list[i][:args.n_samples] for i in range(nclass)]
    labels_list = [labels_list[i][:args.n_samples] for i in range(nclass)]
    # 3. "Re-apilo"
    samples = np.vstack(samples_list)
    print('samples shape', samples.shape)
    labels = np.vstack(labels_list)
    print('labels shape', labels.shape)
    # 4. Normalizo cada muestra de manera que la distancia al átomo más cercano es 1.0 unidades
    for k in range(samples.shape[0]):
        samples[k,...] = samples[k,...]/np.linalg.norm(samples[k][0])
        samples[k,...] = np.array(samples[k,...])
        samples[k,...] = np.nan_to_num(samples[k,...])
    

    # Guardo outputs
    np.save(args.out_name + '_scaled_shuffled_equal_samples.npy', samples)
    
    # tengo array con atomos seleccionados, sus vecinos
    np.save(args.out_name + '_scaled_shuffled_equal_labels.npy', labels)

    samples = []
    samples_list = []
    labels = []
    labels_list = []
    iso = []
    lam = []
    idxs = []
    sel = []
   
   
def get_args():

    #Parse Arguments
    parser = argparse.ArgumentParser(description='Create datasets for pointnet training for crystal structure ID')
    parser.add_argument('--path', help='path to simulations for generating training data', type=str, required=True)
    parser.add_argument('--out_name', help='file prefix to save training dataset', type=str, required=True)
    parser.add_argument('--cutoff', help='neighbor cutoff for point clouds', type=float, required=True)
    parser.add_argument('--max_neigh',help='max neighbors in point clouds',type=int,required=True)
    parser.add_argument('--n_select', help='number training samples to extract from each frame of a simulation', type=int, required=False,default=50)
    parser.add_argument('--n_samples', help='total number training samples per phase', type=int, required=False,default=100000)

    args = parser.parse_args()

    return args 


# Boilerplate notation to run main fxn
if __name__ == "__main__":
   main() 
