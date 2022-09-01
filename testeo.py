import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import MDAnalysis as mda
import argparse
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from sklearn.metrics import confusion_matrix
import multiprocessing
import time
from red_puntos import PointNet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""# Testeo"""



"""Evalúo la red con el enfriamiento de mi sistema. En la simulación de 400 frames (o time steps) se ve la transición de mesofase desordenada a altas temperaturas a mesofase lamelar, para luego llegar a lamelar critalizada a bajas temperaturas"""



# Función modificada de https://github.com/rsdefever/GenStrIde/blob/master/scripts/mda_cluster.py

def evaluo(file_trj,nclass,cutoff,maxneigh):     
    
    u = mda.Universe(file_trj,topology_format='LAMMPSDUMP')
    
    resultados = []


    # Analizo en cada frame de la trayectoria
    for ts in u.trajectory:
        # Genero una lista de vecinos (dentro de las coordenadas especificadas)
        
        nlist = nsgrid.FastNS(cutoff*1.0,u.atoms.positions,ts.dimensions).self_search()

        # Extraigo la información requerida 
        ndxs = nlist.get_indices()
        dxs = nlist.get_dx()
        dists = nlist.get_distances()

        samples = []
        # Preparo las muestras para enviarlas a la red
        for i in range(len(dxs)):
            ## Ordeno los vecinos por distancia (de manera que pueda normalizar
            ## las distancias luego)
            nneigh = int(len(dxs[i])/3)
            np_dxs = np.asarray(dxs[i]).reshape([nneigh,3])
            sort_order = np.asarray(dists[i]).argsort() 
            np_dxs = np_dxs[sort_order]
            if nneigh > 0:
                np_dxs /= np.linalg.norm(np_dxs[0])
            # Corrijo el tamaño del input, sumando o quitando puntos
            if nneigh < maxneigh:
                np_dxs = np.pad(np_dxs,[(0, maxneigh-nneigh), (0, 0)],'constant',)
            elif nneigh > maxneigh:
                np_dxs = np_dxs[:maxneigh]

            # Append sample info
            samples.append(np_dxs)
            
        # Convierto en un array
        np_samples = np.asarray(samples)

        # 
        # cada frame envío a la red
        predictions = PointNet.predigo_con_red(np_samples, steps=len(np_samples))
        predicted_classes = np.argmax(np.rint(predictions), axis=1)
        
        resultados.append(predicted_classes)    # Guardo en lista la predicción sobre
                                                # la clase de cada átomo del 
                                                # sistema
    res = np.asarray(resultados)
    
    print(res.shape)
    
    return res



def main():       

   args = get_args() 

   prueba = evaluo(args.file_trj,args.nclass,args.cutoff,args.maxneigh,args.outname)

   
   return prueba
   
   
   
   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Uses MDAnalysis and PointNet to identify largest cluster of solid-like atoms')
    parser.add_argument('--nclass', help='number of classes', type=int, required=True)
    parser.add_argument('--file_trj', help='path to files', type=str, required=True)
    parser.add_argument('--file_csv', help='path to files', type=str, required=True)
    parser.add_argument('--cutoff', help='neighbor cutoff distance (in nm)', type=float, required=True)
    parser.add_argument('--maxneigh', help='max number of neighbors', type=int, required=True)
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()

    
