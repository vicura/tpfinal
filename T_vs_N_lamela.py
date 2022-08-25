#!/usr/bin/python3

# Lee una salida del TCC y genera varios *.dat con la distribucion de cada uno de los clusters por frame

import sys
import numpy as np
import pandas as pd

def T_vs_N_lamela(archivo_N,archivo_thermo):

   thermo = open(archivo_thermo,'r')
   temp = np.loadtxt(thermo)
   thermo.readline()

   salida = open(input('nombre del archivo de salida: '),'w')
  # salida = input('nombre del archivo de salida: ')
   salida.write("# Temperatura, n_lam, n_lam_ord, n_desord\n")



   with open(archivo_N) as n:
      lines = n.readlines()
#      for j in range(len(temp)):
      k=0 
      for i in range(1, len(lines)):
         if k<200000:
            line = lines[i] 
            if line != '':
               line = line.rstrip('	\n')
               line =  ' '.join(line.split())
               line = line.split(' ')
               n_lam = line[1]
               n_lam_ord = line[2]
               n_desord= line[3]
                                            
          #  for j in range(len(temp)):                                    
               #salida.write(str(int(temp[j][1]))+' '+str(n_lam)+' '+str(n_hex)+'\n')
            #else:
             #  break    
               salida.write(str(int(temp[k][1]))+' '+str(n_lam)+' '+str(n_lam_ord)+' '+str(n_desord)+'\n')
               k += 20
         else:
            break

def main(parametros):
    
   if len(parametros) == 3:
      archivo_N = parametros[1]
      archivo_thermo = parametros[2]
      salida = T_vs_N_lamela(archivo_N,archivo_thermo)

      return salida

if __name__ == '__main__':
    main(sys.argv)
