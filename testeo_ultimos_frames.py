import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import MDAnalysis as mda
import argparse
import nsgrid_rsd as nsgrid
from red_puntos import PointNet
import gc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""# Testeo"""


"""Evalúo la red con el enfriamiento de mi sistema. En la simulación de 400 frames (o time steps) 
se ve la transición de mesofase desordenada a altas temperaturas a mesofase lamelar, para luego 
llegar a lamelar critalizada a bajas temperaturas"""



# Función modificada de https://github.com/rsdefever/GenStrIde/blob/master/scripts/mda_cluster.py

def evaluo(file_trj,nepochs,batch_size,learning_rate,arg,rate,n_classes,cutoff,maxneigh,outname):     
    
    # Cargo red
    net = PointNet(epochs=nepochs,
                   batch_size=batch_size,
                   lr=learning_rate,
                   n_classes=n_classes,
                   arg = arg,
                   rate = rate)
    
    # Cargo topología
    u = mda.Universe(file_trj,topology_format='LAMMPSDUMP')

    # Archivo de salida
    f_summary = pd.DataFrame(columns=['time', 'nº partículas lam', 'nº partículas iso'])
        
    # Analizo los últimos 10 frames de la trayectoria
    for ts in u.trajectory[45:55]:
        
        # Grid based search between positions
       # Searches all the pairs within the initialized coordinates
       # All the pairs among the initialized coordinates are registered in hald the time.
       # Although the algorithm is still the same, but the distance checks can be reduced 
       # to half in this particular case as every pair need not be evaluated twice.
        nlist = nsgrid.FastNS(cutoff*1.0,u.atoms.positions,ts.dimensions).self_search()

        

        # Extraigo la información requerida 
        ndxs = nlist.get_indices()        # Individual neighbours of query atom.
                                          # For every queried atom ``i``, an array of all its neighbors
                                          # indices can be obtained from ``get_indices()[i]``
        dxs = nlist.get_dx()              # Devuelve coordenadas de los vecinos?
                                          # For every queried atom ``i``, an array of all its neighbors
                                          # coordinates can be obtained from ``get_dx()[i]``
        dists = nlist.get_distances()     # Distance corresponding to individual neighbors of query atom
                                          # For every queried atom ``i``, a list of all the distances
                                          # from its neighboring atoms can be obtained from 
                                          # ``get_distances()[i]``.

        samples = []
        
        # Preparo las muestras para enviarlas a la red

        # Itero sobre particulas
        for i in range(len(dxs)):
            ## Ordeno los vecinos por distancia (de manera que pueda normalizar
            ## las distancias luego)
            nneigh = int(len(dxs[i])/3)   # Obtengo numero de vecinos
                                          # por que divido por 3? 3 dimensiones?
            np_dxs = np.asarray(dxs[i]).reshape([nneigh,3])  # transformo en array la lista de coord 
                                                             # de vecino y hago cambio de forma
            sort_order = np.asarray(dists[i]).argsort()  # Returns the indices that would sort an array.
            np_dxs = np_dxs[sort_order]                  # Ordeno array de vecinos de menor a mayor distancia
            
            # Normalizo distancias
            if nneigh > 0:
                np_dxs /= np.linalg.norm(np_dxs[0]) # equivalente a np_dxs = np_dxs/np.linalg.norm(np_dxs[0])
                                                    # Normalizo de manera que la distancia al átomo más cercano es 1.0 unidades
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
        print('samples shape: ',np_samples.shape)
        
        input_shape = (maxneigh, n_classes, 1)
        # cada frame envío a la red
        predictions = net.predigo_con_red(arg=arg,rate=rate, n_classes= n_classes, input_shape=input_shape, 
        samples=np_samples, steps=len(np_samples))
        
        predicted_classes = np.argmax(np.rint(predictions), axis=1)
        print('predicted classes: ',predicted_classes)
        
        
        np_samples = []
        samples = []
        sys.stdout.flush()
        
        
        # Extract different atom types
        lam_atoms = np.where(predicted_classes == 0)[0]
        #print('lam atoms :',lam_atoms.shape[0])
        iso_atoms = np.where(predicted_classes == 1)[0]
        #print('iso atoms :',iso_atoms.shape[0])
        #f_summary.write("{:8.3f}{:8d}{:8d}\n".format(ts.time,lam_atoms.shape[0],iso_atoms.shape[0]))
        new_row = pd.Series({'time':ts.time, 'nº partículas lam': lam_atoms.shape[0], 
                            'nº partículas iso':iso_atoms.shape[0]})
        f_summary = pd.concat([f_summary, new_row.to_frame().T], ignore_index=True)
        
                                                                                                
                                                                                       
    with open(outname+'_por_frame.txt', 'w') as f:
       new_df = f_summary.to_string(index=False)
       f.write(new_df)
    
    promedios = pd.DataFrame(columns=['cutoff', 'nº partículas lam','nº partículas iso'])
    new_row = pd.Series({'cutoff':cutoff, 'nº partículas lam':f_summary['nº partículas lam'].mean() , 
                            'nº partículas iso':f_summary['nº partículas iso'].mean()})
    promedios = pd.concat([promedios, new_row.to_frame().T], ignore_index=True)
    


#.tail(10)
    with open(outname+'_promedios.txt', 'w') as f:
       new_df = promedios.to_string(index=False)
       f.write(new_df)   
    
    gc.collect()
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

    
