import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle

import multiprocessing
import time



"""# Testeo"""



"""Evalúo la red con el enfriamiento de mi sistema. En la simulación de 400 frames (o time steps) se ve la transición de mesofase desordenada a altas temperaturas a mesofase lamelar, para luego llegar a lamelar critalizada a bajas temperaturas"""



# Función modificada de https://github.com/rsdefever/GenStrIde/blob/master/scripts/mda_cluster.py

def main(path, nclass, cutoff, maxneigh, outname):

    u = mda.Universe(path,topology_format='LAMMPSDUMP')
    
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

        # En cada frame envío a la red
        predictions = PointNet.predict(np_samples, steps=len(np_samples))
        predicted_classes = np.argmax(np.rint(predictions), axis=1)
        
        resultados.append(predicted_classes)    # Guardo en lista la predicción sobre
                                                # la clase de cada átomo del 
                                                # sistema

    return np.asarray(resultados)


def grafico_resultados(prueba,archivo_dat,archivo_csv):

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
     class_lam.append(lam/2048)                        # Divido por el número de 
     class_lam_ord.append(lam_ord/2048)                # partículas totales para
     class_lam_desord.append(lam_desord/2048)          # tener la fracción de 
                                                       # partículas por frame de 
                                                       # cada clase

   frame = [n for n,i in enumerate(prueba)]
   print(frame)

   with open(archivo_dat, 'r') as in_file:
      with open(archivo_csv, 'w') as out_file:
         in_file.readline()
         for i in range(10000000):
            f = in_file.readline()
            if f != '':
                  f = f.rstrip('        \n')
                  f = f.replace('   ',',')
                  f = f.split(',')
                  f = list(filter(('').__ne__, f))
                  f = ",".join(map(str, f))           
                  out_file.write(f)
                  out_file.write("\n")
           else:
                  break
   data = pd.read_csv(archivo_csv,skiprows=1,sep=',',header=None,names=['Step','KEng','T','Volume'])
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
   fig.show()
   
   return

def main(parametros):
     
   archivo_lammpstrj = parametros[1]
   
   archivo_dat = parametros[2]
   
   archivo_csv = parametros[3]

   prueba = pre_eval(archivo_lammpstrj,3,2.0,50,prueba)
   
   grafico = grafico_resultados(prueba,archivo_dat,archivo_csv)
   
   return 
   
   
   
   
if __name__ == "__main__":
   with Pool() as pool:
      pool.map(main(), range(8))

    
