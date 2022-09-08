import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import MDAnalysis as mda
import argparse
import nsgrid_rsd as nsgrid
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

def evaluo(file_trj,nepochs,batch_size,learning_rate,arg,rate,n_classes,cutoff,maxneigh,outname):     
    
    net = PointNet(epochs=nepochs,
                   batch_size=batch_size,
                   lr=learning_rate,
                   n_classes=n_classes,
                   arg = arg,
                   rate = rate)
    
    
    u = mda.Universe(file_trj,topology_format='LAMMPSDUMP')
    
    #resultados = []

    # File to write output
    f_summary = open(outname+'_summary.mda','w')
    f_class = open(outname+'_class.mda','w')
    
    f_summary.write("# Time, n_lam, n_lam_ord, n_desord\n")
    
        
    # Analizo en cada frame de la trayectoria
    for ts in u.trajectory:
        
        f_class.write("ITEM: TIMESTEP\n")
        f_class.write("{:d}\n".format(ts.frame))
        f_class.write("ITEM: NUMBER OF ATOMS\n")
        f_class.write("{:d}\n".format(ts.n_atoms))
        f_class.write("ITEM: BOX BOUNDS pp pp pp\n")
        f_class.write("-"+"{0:.16e} {0:.16e}\n".format(1.0453842000444242e+01,1.0453842000444242e+01))
        f_class.write("-"+"{0:.16e} {0:.16e}\n".format(1.0453842000444242e+01,1.0453842000444242e+01))
        f_class.write("-"+"{0:.16e} {0:.16e}\n".format(1.0453842000444242e+01,1.0453842000444242e+01))
        f_class.write("ITEM: ATOMS id type mol x y z\n")
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
        
       
        a,b,c = np_samples.shape
        np_samples = np_samples.reshape(a,b,c,-1) 
        print(np_samples.shape)
        
        input_shape = (maxneigh, n_classes, 1)
        # cada frame envío a la red
        predictions = net.predigo_con_red(arg=arg,rate=rate, n_classes= n_classes, input_shape=input_shape, 
        samples=np_samples, steps=len(np_samples))
        #predicted_classes = np.asarray(predictions)
        #print(predicted_classes)
        
        predicted_classes = np.argmax(np.rint(predictions), axis=1)
        print(predicted_classes)
        
        
        np_samples = []
        samples = []
        
        
        
        # Extract different atom types
        lam_atoms = np.where(predicted_classes == 0)[0]
        lam_ord_atoms = np.where(predicted_classes == 1)[0]
        desord_atoms = np.where(predicted_classes == 2)[0]

        
        f_summary.write("{:8.3f}{:8d}{:8d}{:8d}\n".format(ts.time,lam_atoms.shape[0],lam_ord_atoms.shape[0],desord_atoms.shape[0]))
        
        for atom in u.atoms:
         #  if  predicted_classes[atom.index] == 2:
              f_class.write("{:d} {:s} {:d} {:.10f} {:.10f} {:.10f}\n".format(atom.index,atom.type,predicted_classes[atom.index],atom.position[0],atom.position[1],atom.position[2]))  
                                                                                       
                                                                                       
    f_summary.close()
    f_class.close()

        
        #resultados.append(predicted_classes)    # Guardo en lista la predicción sobre
                                                # la clase de cada átomo del 
                                                # sistema
    #res = np.asarray(resultados)
    #np.save(outname + 'npy', res)
    #print(res.shape)
    
    return 


def main():       

   args = get_args() 
                   
   prueba = evaluo(args.file_trj,args.nepochs,args.batch_size,args.learning_rate,args.arg,
   args.rate,args.n_classes,args.cutoff,args.maxneigh,args.outname)

   
   return prueba
   
   
   
   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Uses MDAnalysis and PointNet to identify largest cluster of solid-like atoms')
    parser.add_argument('--n_classes', help='number of classes', type=int, required=True)
    parser.add_argument('--file_trj', help='path to files', type=str, required=True)
    parser.add_argument('--cutoff', help='neighbor cutoff distance (in nm)', type=float, required=True)
    parser.add_argument('--rate', help='rate of dropout', type=float, required=False, default=0.3)
    parser.add_argument('--learning_rate',help='learning rate',type=float, required=False, default=0.001)
    parser.add_argument('--nepochs', help='number of epochs for training', type=int, required=False, default=15)
    parser.add_argument('--batch_size', help='size of batch for training', type=int, required=False, default=32)
    parser.add_argument('--arg',help='argument',type=float, required=False, default=1e-5)
    parser.add_argument('--maxneigh', help='max number of neighbors', type=int, required=True)
    parser.add_argument('--outname', help='name output file', type=str, required=True)    
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()

    
