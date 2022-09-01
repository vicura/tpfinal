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




def grafico_resultados(prueba,archivo_csv,atoms,outname):

   
   class_lam = []
   class_lam_ord = []
   class_lam_desord = []

   for i in prueba:
     lam = 0
     lam_ord = 0
     lam_desord = 0
     for j in i:
       if j == 0:
         lam += 1
       if j == 1:
         lam_ord += 1 
       if j == 2:
         lam_desord += 1
     class_lam.append(lam/atoms)                        # Divido por el número de 
     class_lam_ord.append(lam_ord/atoms)                # partículas totales para
     class_lam_desord.append(lam_desord/atoms)          # tener la fracción de 
                                                        # partículas por frame de 
                                                        # cada clase

   frame = [n for n,i in enumerate(prueba)]
   print(frame)


   data = pd.read_csv(archivo_csv,sep=',',header=None,names=['Step','T','TotEng','PotEng','KinEng',
   'E_pair','E_bond','Volume','Press','Densidad'])
   
   print(data)

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
   fig.write_image(args.outname+".png")
   return
   


def main():       

   args = get_args() 

   res = grafico_resultados(args.file_resultados,args.archivo_csv,args.atoms,args.outname)
   
   return res
   
   
   
   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Uses MDAnalysis and PointNet to identify largest cluster of solid-like atoms')
    parser.add_argument('--file_resultados', help='array with results from pointnet', type=int, required=True)
    parser.add_argument('--file_csv', help='path to file csv', type=str, required=True)
    parser.add_argument('--atoms', help='number of atoms in system', type=int, required=True)
    parser.add_argument('--outname', help='name output file', type=str, required=True)
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()    
   
