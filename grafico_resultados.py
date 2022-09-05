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




def grafico_resultados(summary,archivo_csv,atoms,outname):
   
   
   class_lam = []
   class_lam_ord = []
   class_lam_desord = []

   #cargo data
   with open(summary) as n:
      lines = n.readlines()
#      for j in range(len(temp)):
      #k=0 
      for i in range(1, len(lines)):
         #if k<200000:
            line = lines[i] 
            if line != '':
               line = line.rstrip('	\n')
               line =  ' '.join(line.split())
               line = line.split(' ')
               n_lam = line[1]
               class_lam.append(n_lam/atoms)
               n_lam_ord = line[2]
               class_lam_ord.append(n_lam_ord/atoms)
               n_desord= line[3]
               class_lam_desord.append(n_lam_desord/atoms)                              
  
               #salida.write(str(int(temp[k][1]))+' '+str(n_lam/atoms)+' '+str(n_lam_ord/atoms)+' '+str(n_desord/atoms)+'\n')
               #k += 1
         #else:
          #  break
   
   


   #frame = [n for n,i in enumerate(data)]
   #print(frame)


   data = pd.read_csv(archivo_csv,sep=',',header=None,names=['Step','T','TotEng','PotEng','KinEng',
   'E_pair','E_bond','Volume','Press','Densidad'])
   
   #print(data)

   ml = pd.DataFrame({'lamelar': class_lam[:-1], 
                   'lamelar ordenado': class_lam_ord[:-1], 
                   'lamelar desordenado' : class_lam_desord[:-1],
                   'temperatura': data['T'],
                   'volumen': data['Volume']})

   print(ml)


   fig = make_subplots(specs=[[{"secondary_y": True}]])
   fig.add_trace(
      go.Scatter(x=ml['temperatura'], y=ml['volumen'],
                 mode='markers',
                 name='quenching/enfriamiento<br>lamelar',
                 marker_color = "rgb(215,48,39)"),
      secondary_y=False,
      )
   fig.add_trace(
       go.Scatter(x=ml['temperatura'],y=ml['lamelar'], mode='markers',
               name='lamelar',
               marker_color ="rgb(253,174,97)"),
               secondary_y=True,
       )
   fig.add_trace(
       go.Scatter(x=ml['temperatura'],y=ml['lamelar ordenado'], mode='markers',
               name='lamelar ordenado',
               marker_color ="rgb(116,173,209)"),
               secondary_y=True,
       )
    fig.add_trace(
       go.Scatter(x=ml['temperatura'],y=ml['lamelar desordenado'], mode='markers',
               name='fase isotrópica',
               marker_color ="rgb(49,54,149)"),
               secondary_y=True,
       )
   fig.update_traces(marker_size=4)
   # Set x-axis title
   fig.update_xaxes(title_text="Temperatura",range=[50,550])
   fig.update_yaxes(title_text="Volumen", secondary_y=False)
   fig.update_yaxes(title_text="Fracción de partículas", secondary_y=True)
   fig.update_layout(
    autosize=False,
    width=1000,
    height=800,legend=dict(
    yanchor="top",
    y=0.6,
    xanchor="left",
    x=0.001,
    font=dict(
            size=12,
            color="black"
        )))
   #fig.show()
   fig.write_image(outname+".png")
   return
   


def main():       

   args = get_args() 

   res = grafico_resultados(args.file_summary,args.archivo_csv,args.atoms,args.outname)
   
   return res
   
   
   
   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Uses MDAnalysis and PointNet to identify largest cluster of solid-like atoms')
    parser.add_argument('--file_summary', help='array with results from pointnet', type=int, required=True)
    parser.add_argument('--file_csv', help='path to file csv', type=str, required=True)
    parser.add_argument('--atoms', help='number of atoms in system', type=int, required=True)
    parser.add_argument('--outname', help='name output file', type=str, required=True)
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()    
   
